import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from typing import Dict, Optional
import io
import base64
import tqdm
import json
from pathlib import Path

# Optional imports for training (not needed for CLI plotting)
import torch
import wandb


class LipschitzAnalyzer:
    """
    Analyzer for checking the assumption:
    | ∇ loss(x) - ∇ loss(y) |_* ≤ (K_0 + K_1 * (f(x) - f(x*)) + K_ρ * (f(x) - f(x*))^ρ) * | x - y |

    where:
    - x, y are current and previous model weights
    - | |_* and | | are dual norms (using Frobenius for now)
    - f(x*) is the optimal loss value
    - K_0, K_1, ρ, K_ρ are parameters to be fitted
    """

    def __init__(
        self,
        enabled: bool = False,
        weight_norm_type: str = 'fro',
        rho: float = 2,
        f_star: float = 1.5,
        fit_rho: bool = True,
        results_dir: Optional[str] = None
    ):
        self.enabled = enabled
        if not enabled:
            return

        self.data_points = []  # List of (grad_diff_norm, loss_val, weight_diff_norm)
        self.prev_weights = None
        self.prev_grads = None
        self.rho = rho
        self.f_star = f_star
        self.fit_rho = fit_rho
        self.results_dir = Path(results_dir) if results_dir else None
        if weight_norm_type == "fro":
            self.weight_norm_type = "fro"
            self.grad_norm_type = "fro*"
        elif weight_norm_type == "muon":
            self.weight_norm_type = "muon"
            self.grad_norm_type = "muon*"
        elif weight_norm_type == "signum":
            self.weight_norm_type = "signum"
            self.grad_norm_type = "signum*"
        else:
            raise ValueError(f"Unsupported weight norm type: {weight_norm_type}")

    def is_enabled(self, iteration) -> bool:
        enabled = self.enabled # and (self.min_analysis_steps <= iteration <= self.max_analysis_steps)
        return enabled

    def _get_model_weights_flat(self, model):
        """Get flattened model weights"""
        weights = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                weights.append(param.data.clone().detach())
        return weights

    def _get_model_grads_flat(self, model):
        """Get flattened model gradients"""
        grads = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.clone().detach())
        return grads

    def _norm(self, tensors, type: str = 'fro') -> float:
        """Compute norm over an iterable of tensors."""
        if type == "muon":
            return max(torch.linalg.norm(t, ord=2) for t in tensors if t.dim() == 2)
        elif type == "muon*":
            return sum(torch.linalg.norm(t, ord="nuc") for t in tensors if t.dim() == 2)
        elif type == "signum":
            return max(torch.linalg.vector_norm(t, ord=float('inf')) for t in tensors if t.dim() == 2)
        elif type == "signum*":
            return sum(torch.linalg.vector_norm(t, ord=1) for t in tensors if t.dim() == 2)
        elif type == "fro":
            return max(torch.linalg.norm(t, ord="fro") for t in tensors if t.dim() == 2)
        elif type == "fro*":
            return sum(torch.linalg.norm(t, ord="fro") for t in tensors if t.dim() == 2)
        else:
            raise ValueError(f"Unsupported norm type: {type}!!!")


    def update_with_grads(
        self,
        prev_grads: "torch.Tensor",
        current_grads: "torch.Tensor",
        prev_weights: "torch.Tensor",
        current_weights: "torch.Tensor",
        loss_val: float,
        iteration: int
    ):
        """
        Update analyzer with pre-computed gradients and weights
        This is used when gradients are computed on the same batch before and after opt.step()

        Args:
            prev_grads: Gradients before opt.step() (on same batch)
            current_grads: Gradients after opt.step() (on same batch)
            prev_weights: Weights before opt.step()
            current_weights: Weights after opt.step()
            loss_val: Loss value
            iteration: Current iteration number
        """
        if prev_grads is None or current_grads is None:
            return

        # Compute | ∇ loss(x) - ∇ loss(y) |_* and | x - y |
        # grad_diff = current_grads - prev_grads
        grad_diff_norm = self._norm(
            (cg - pg for cg, pg in zip(current_grads, prev_grads)),
            type=self.grad_norm_type
        )

        weight_diff_norm = self._norm(
            (cw - pw for cw, pw in zip(current_weights, prev_weights)),
            type=self.weight_norm_type
        )

        # Store data point
        if weight_diff_norm > 1e-12:  # Avoid division by zero
            data_point = {
                'grad_diff_norm': grad_diff_norm.item(),
                'loss_val': loss_val - self.f_star,
                'weight_diff_norm': weight_diff_norm.item(),
                'iteration': iteration
            }
            self.data_points.append(data_point)

            # Log to wandb
            if wandb.run:
                wandb.log({
                    'lipschitz/grad_diff_norm': grad_diff_norm,
                    'lipschitz/loss_val': loss_val - self.f_star,
                    'lipschitz/weight_diff_norm': weight_diff_norm,
                    'lipschitz/ratio': grad_diff_norm / weight_diff_norm,
                    'lipschitz/loss': loss_val - self.f_star,
                    'iter': iteration
                })

    def _fit_least_squares(
        self, grad_diff_norms: np.ndarray,
        loss_vals: np.ndarray,
        weight_diff_norms: np.ndarray,
        rho: float
    ) -> Optional[Dict[str, float]]:
        """
        Fit K_0, K_1, K_ρ using least squares for the linear relationship:
        ratio = K_0 + K_1 * loss + K_ρ * loss^ρ
        where ratio = |∇loss(x) - ∇loss(y)|_* / |x - y|
        """
        try:
            # Compute Lipschitz ratios
            ratios = grad_diff_norms / weight_diff_norms

            # Create design matrix for linear regression: [1, loss, loss^rho]
            loss_rho = loss_vals ** rho
            X = np.column_stack([np.ones(len(loss_vals)), loss_vals, loss_rho])
            # X = np.column_stack([np.ones(len(loss_vals)), loss_rho])
            y = ratios

            # Solve least squares: X * [K_0, K_1, K_ρ]ᵀ = y
            params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            K_0, K_1, K_rho = params
            # K_0, K_rho = params

            # Calculate R-squared for goodness of fit
            y_pred = K_0 + K_1 * loss_vals + K_rho * loss_rho
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'K_0': float(K_0),
                'K_1': float(K_1),
                'rho': float(rho),
                'K_rho': float(K_rho),
                'r_squared': float(r_squared),
                'num_data_points': len(ratios)
            }
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return None

    def fit_parameters(self) -> Optional[Dict[str, float]]:
        """
        Fit K_0, K_1, K_ρ parameters using least squares

        Returns:
            Dictionary with fitted parameters or None if insufficient data
        """
        # Extract data
        grad_diff_norms = np.array([dp['grad_diff_norm'] for dp in self.data_points])
        loss_vals = np.array([dp['loss_val'] for dp in self.data_points])
        weight_diff_norms = np.array([dp['weight_diff_norm'] for dp in self.data_points])

        # Filter out any invalid data
        valid_mask = (grad_diff_norms > 0) & (loss_vals >= 0) & (weight_diff_norms > 0)
        grad_diff_norms = grad_diff_norms[valid_mask]
        loss_vals = loss_vals[valid_mask]
        weight_diff_norms = weight_diff_norms[valid_mask]

        if len(grad_diff_norms) < 5:
            return None

        if not self.fit_rho:
            # Use fixed rho
            return self._fit_least_squares(grad_diff_norms, loss_vals, weight_diff_norms, self.rho)

        # Grid search for optimal rho in range (1, 4] with smart sampling around rho_0 = 2

        # Smart sampling: denser around rho=2, sparser at extremes
        rho_candidates = []

        # Dense sampling around rho = 2
        rho_candidates.extend(np.linspace(1.05, 50, 100))

        # # Medium sampling
        # rho_candidates.extend(np.linspace(1.2, 1.6, 9))   # medium density
        # rho_candidates.extend(np.linspace(2.4, 3.0, 13))  # medium density

        # # Sparse sampling at extremes
        # rho_candidates.extend(np.linspace(1.05, 1.15, 6))  # at lower end
        # rho_candidates.extend(np.linspace(3.1, 4.0, 10))   # at upper end

        # # Remove duplicates and ensure all are > 1
        # rho_candidates = sorted([rho for rho in set(rho_candidates) if rho > 1.0])

        best_params = None
        best_r_squared = -1

        for rho in tqdm.tqdm(rho_candidates, desc="Fitting rho"):
            params = self._fit_least_squares(grad_diff_norms, loss_vals, weight_diff_norms, rho)
            if params and params['r_squared'] > best_r_squared:
                best_r_squared = params['r_squared']
                best_params = params

        return best_params

    def plot_results(self, fitted_params: Dict[str, float]) -> Optional[str]:
        """
        Create visualization of the results

        Args:
            fitted_params: Dictionary with fitted K_0, K_1, ρ, K_ρ

        Returns:
            Base64 encoded image string or None if failed
        """
        try:
            print(f"Creating plot with {len(self.data_points)} data points...")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract data
            grad_diff_norms = np.array([dp['grad_diff_norm'] for dp in self.data_points])
            loss_vals = np.array([dp['loss_val'] for dp in self.data_points])  # loss(x) - loss(x*)
            weight_diff_norms = np.array([dp['weight_diff_norm'] for dp in self.data_points])
            iterations = np.array([dp['iteration'] for dp in self.data_points])

            # Main scatter plot - loss(x) vs ratio with color mapping by iteration
            ratios = grad_diff_norms / weight_diff_norms  # |∇loss(x) - ∇loss(y)|_* / |x - y|

            # Create color map: blue (early) to red (late iterations)
            scatter = ax.scatter(loss_vals, ratios, c=iterations, cmap='coolwarm',
                               alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Iteration', fontsize=10)

            ax.set_xlabel(f'loss(x) - loss(x*) = loss(x) - {self.f_star}', fontsize=12)
            ax.set_ylabel('||∇loss(x) - ∇loss(y)||_* / ||x - y||', fontsize=12)
            ax.set_title('Lipschitz Analysis: Data vs Fitted Line', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Plot fitted line
            if fitted_params:
                K_0, K_1, K_rho = fitted_params['K_0'], fitted_params['K_1'], fitted_params['K_rho']
                rho = fitted_params['rho']
                r_squared = fitted_params.get('r_squared', 0)
                loss_range = np.linspace(loss_vals.min(), loss_vals.max(), 100)
                fitted_line = K_0 + K_1 * loss_range + K_rho * loss_range**rho  # K_0 + K_1 * loss + K_ρ * loss^ρ

                fit_status = "fitted" if self.fit_rho else "fixed"
                ax.plot(loss_range, fitted_line, 'r-', linewidth=3,
                        label=f'Fitted line: K_0 + K_1·loss + K_ρ·loss^{rho:.2f}\nR2 score = {r_squared:.3f}')
                        #\nK_0={K_0:.2f}, K_1={K_1:.2f}, K_ρ={K_rho:.2f}\nρ={rho:.2f} ({fit_status}), R²={r_squared:.3f}')

                # Add some visual validation
                fitted_vals = K_0 + K_1 * loss_vals + K_rho * loss_vals**rho
                violations = np.sum(ratios > fitted_vals)
                total_points = len(ratios)
                violation_pct = 100 * violations / total_points
                # ax.text(0.02, 0.98, f'Points above line: {violations}/{total_points} ({violation_pct:.1f}%)',
                #        transform=ax.transAxes, verticalalignment='top',
                #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                # threshold = (K_0 / (K_rho * (rho - 1)))**rho
                # ax.text(0.02, 0.07, f'Estimated threshold: {threshold:.4f}',
                #        transform=ax.transAxes, verticalalignment='top',
                #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.legend(fontsize=11, loc='upper right')

            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            print("Plot generated successfully!")
            return image_base64

        except Exception as e:
            print("Error creating plot:", e)
            import traceback
            traceback.print_exc()
            return None

    def save_data(self, filepath: Path):
        """Save collected data points to JSON file"""
        data_to_save = {
            'data_points': self.data_points,
            'config': {
                'rho': self.rho,
                'f_star': self.f_star,
                'fit_rho': self.fit_rho,
                'weight_norm_type': self.weight_norm_type,
                'grad_norm_type': self.grad_norm_type,
            }
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Saved Lipschitz analysis data to {filepath}")

    def finalize_analysis(self):
        """
        Perform final analysis and logging at the end of training
        """

        print("\n=== Lipschitz Analysis Results ===")
        print(f"Collected {len(self.data_points)} data points")

        # Save data to file if results directory is specified
        if self.results_dir and len(self.data_points) > 0:
            data_file = self.results_dir / "lipschitz_analysis_data.json"
            self.save_data(data_file)

        if len(self.data_points) > 0:
            # Show summary statistics
            grad_norms = [dp['grad_diff_norm'] for dp in self.data_points]
            weight_norms = [dp['weight_diff_norm'] for dp in self.data_points]
            losses = [dp['loss_val'] for dp in self.data_points]

            print(f"Gradient diff norm range: [{min(grad_norms):.6e}, {max(grad_norms):.6e}]")
            print(f"Weight diff norm range: [{min(weight_norms):.6e}, {max(weight_norms):.6e}]")
            print(f"Loss range: [{min(losses):.6f}, {max(losses):.6f}]")

        if len(self.data_points) < 10:
            print("Insufficient data points for parameter fitting (need at least 10)")
            if len(self.data_points) > 0:
                print("Try increasing --iterations or decreasing --log_interval")
            return

        # Fit parameters
        fitted_params = self.fit_parameters()

        if fitted_params:
            fit_status = "fitted" if self.fit_rho else "fixed"
            print(f"Fitted parameters (ρ = {fitted_params['rho']:.3f} {fit_status}):")
            print(f"  K_0 = {fitted_params['K_0']:.6e}")
            print(f"  K_1 = {fitted_params['K_1']:.6e}")
            print(f"  ρ = {fitted_params['rho']:.3f} ({fit_status})")
            print(f"  K_ρ = {fitted_params['K_rho']:.6e}")
            print(f"  R² = {fitted_params.get('r_squared', 0):.6f}")

            # Validation check
            grad_norms = [dp['grad_diff_norm'] for dp in self.data_points]
            weight_norms = [dp['weight_diff_norm'] for dp in self.data_points]
            losses = [dp['loss_val'] for dp in self.data_points]

            ratios = np.array(grad_norms) / np.array(weight_norms)
            fitted_vals = fitted_params['K_0'] + fitted_params['K_1'] * np.array(losses) + fitted_params['K_rho'] * np.array(losses)**fitted_params['rho']
            violations = np.sum(ratios > fitted_vals)
            print(f"  Validation: {violations}/{len(ratios)} points above bound ({100*violations/len(ratios):.1f}%)")

            # Log to wandb
            if wandb.run:
                wandb.log({
                    'lipschitz/final_K_0': fitted_params['K_0'],
                    'lipschitz/final_K_1': fitted_params['K_1'],
                    'lipschitz/final_rho': fitted_params['rho'],
                    'lipschitz/final_K_rho': fitted_params['K_rho'],
                    'lipschitz/final_r_squared': fitted_params['r_squared']
                })

                # Create and log plot
                print("Attempting to create visualization plot...")
                plot_image = self.plot_results(fitted_params)
                if plot_image:
                    print("Plot created successfully, logging to W&B...")
                    try:
                        # Log image to wandb
                        try:
                            from PIL import Image
                        except ImportError:
                            import PIL.Image as Image

                        image_data = base64.b64decode(plot_image)
                        image = Image.open(io.BytesIO(image_data))
                        if wandb.run:
                            wandb.log({"lipschitz/analysis_plot": wandb.Image(image)})
                            print("Plot successfully logged to W&B!")
                        else:
                            print("W&B not available, skipping W&B logging")
                    except Exception as e:
                        print(f"Error logging plot to W&B: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("Failed to create plot - check matplotlib and PIL installation")
        else:
            print("Failed to fit parameters")


def load_and_plot(data_path: str, output_path: Optional[str] = None,
    min_analysis_steps: int = 0, max_analysis_steps: int = -1, max_fit_steps: int = -1):
    """
    Load saved Lipschitz analysis data and create plots

    Args:
        data_path: Path to JSON file with saved data
        output_path: Optional path to save the plot image
    """
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    # Load data
    with open(f"{data_path}/lipschitz_analysis_data.json", 'r') as f:
        saved_data = json.load(f)

    data_points = saved_data['data_points']
    mx_len = len(data_points) if max_analysis_steps == -1 else max_analysis_steps
    data_points = data_points[min_analysis_steps:mx_len]
    config = saved_data['config']

    print(f"\nLoaded {len(data_points)} data points")
    print(f"Configuration: {config}")
    
    # Filter outliers using 0.01 and 0.99 percentiles of ratios
    grad_norms = np.array([dp['grad_diff_norm'] for dp in data_points])
    weight_norms = np.array([dp['weight_diff_norm'] for dp in data_points])
    ratios = grad_norms / weight_norms
    low, high = np.percentile(ratios, [1, 99])
    filtered = [dp for dp, r in zip(data_points, ratios) if low <= r <= high]
    print(f"Filtered outliers: {len(data_points) - len(filtered)} removed, {len(filtered)} remain")
    data_points = filtered
    
    if len(data_points) < 10:
        print("Insufficient data points for analysis (need at least 10)")
        return

    # Create analyzer instance to use its methods
    analyzer = LipschitzAnalyzer(
        enabled=True,
        rho=config['rho'],
        f_star=config['f_star'],
        fit_rho=config['fit_rho']
    )
    mx_anal_len = len(data_points) if max_fit_steps == -1 else max_fit_steps
    analyzer.data_points = data_points[:mx_anal_len]

    # Fit parameters
    print("\nFitting parameters...")
    fitted_params = analyzer.fit_parameters()

    if fitted_params:
        fit_status = "fitted" if config['fit_rho'] else "fixed"
        print(f"\nFitted parameters (ρ = {fitted_params['rho']:.3f} {fit_status}):")
        print(f"  K_0 = {fitted_params['K_0']:.6e}")
        print(f"  K_1 = {fitted_params['K_1']:.6e}")
        print(f"  ρ = {fitted_params['rho']:.3f} ({fit_status})")
        print(f"  K_ρ = {fitted_params['K_rho']:.6e}")
        print(f"  R² = {fitted_params.get('r_squared', 0):.6f}")

        # Validation check
        grad_norms = np.array([dp['grad_diff_norm'] for dp in data_points])
        weight_norms = np.array([dp['weight_diff_norm'] for dp in data_points])
        losses = np.array([dp['loss_val'] for dp in data_points])

        ratios = grad_norms / weight_norms
        fitted_vals = fitted_params['K_0'] + fitted_params['K_1'] * losses + fitted_params['K_rho'] * losses**fitted_params['rho']
        violations = np.sum(ratios > fitted_vals)
        print(f"  Validation: {violations}/{len(ratios)} points above bound ({100*violations/len(ratios):.1f}%)")

        # Create plot
        print("\nCreating plot...")
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(20, 6))

        iterations = np.array([dp['iteration'] for dp in data_points])

        # Main scatter plot with color mapping by iteration
        scatter = ax.scatter(losses, ratios, c=iterations, cmap='coolwarm',
                           alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Training Iteration', fontsize=10)

        ax.set_xlabel(f'loss(x) - loss(x*) = loss(x) - {config["f_star"]}', fontsize=12)
        ax.set_ylabel('||∇loss(x) - ∇loss(y)||_* / ||x - y||', fontsize=12)
        ax.set_title('Lipschitz Analysis: Data vs Fitted Line', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Plot fitted line
        K_0, K_1, K_rho = fitted_params['K_0'], fitted_params['K_1'], fitted_params['K_rho']
        rho = fitted_params['rho']
        r_squared = fitted_params.get('r_squared', 0)
        losses_anal = np.array([dp['loss_val'] for dp in analyzer.data_points])
        loss_range = np.linspace(losses_anal.min(), losses_anal.max(), 100)
        fitted_line = K_0 + K_1 * loss_range + K_rho * loss_range**rho

        ax.plot(loss_range, fitted_line, 'r-', linewidth=3,
                label=f'Fitted line: K_0 + K_1·loss + K_ρ·loss^{rho:.2f}\nR2 score = {r_squared:.3f}')
                #\nK_0={K_0:.2f}, K_1={K_1:.2f}, K_ρ={K_rho:.2f}\nρ={rho:.2f} ({fit_status}), R²={r_squared:.3f}')

        # # Add statistics
        # violation_pct = 100 * violations / len(ratios)
        # ax.text(0.02, 0.98, f'Points above line: {violations}/{len(ratios)} ({violation_pct:.1f}%)',
        #        transform=ax.transAxes, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # threshold = (K_0 / (K_rho * (rho - 1)))**rho if rho > 1 else 0
        # ax.text(0.02, 0.07, f'Estimated threshold: {threshold:.4f}',
        #        transform=ax.transAxes, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(fontsize=15)
        
        
        ax1.scatter(iterations, losses / ratios, c=iterations, cmap='coolwarm',
                   alpha=0.7, s=30, edgecolors='black', linewidth=0.2)
        ax1.set_xlabel('Training Iteration (t)', fontsize=12)
        ax1.set_ylabel(r'lr($\Delta_t$)', fontsize=12)
        ax1.set_title('Theoretical Learning rate vs Training Iteration', fontsize=14)
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('Training Iteration', fontsize=10)
        
        iterations_anal = np.array([dp['iteration'] for dp in analyzer.data_points])
        losses_anal = np.array([dp['loss_val'] for dp in analyzer.data_points])
        fitted_line = (losses_anal) / (K_0 + K_1 * losses_anal + K_rho * losses_anal**rho)
        
        ax1.plot(iterations_anal, fitted_line, 'r-', linewidth=3,
                label=f'Fitted line')
        
        
        
        ax1.legend(fontsize=15)

        plt.tight_layout()
        # Save plot
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
        else:
            # Save to same directory as data file
            output_path = data_path / f"lipschitz_analysis_plot_min={min_analysis_steps}_max={max_analysis_steps}_anal_max={max_fit_steps}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        plt.close()
    else:
        print("Failed to fit parameters")

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from main import get_exp_name, get_args

    args, parser = get_args()
    run_name = get_exp_name(args, parser)

    min_analysis_steps, max_analysis_steps, max_fit_steps = 150, 19000, 500

    load_and_plot(f"lip_points/{run_name}", None, 
                  min_analysis_steps, max_analysis_steps, max_fit_steps)
