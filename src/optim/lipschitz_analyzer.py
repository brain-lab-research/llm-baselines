import torch
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from typing import Dict, Optional
import io
import base64


class LipschitzAnalyzer:
    """
    Analyzer for checking the assumption:
    | ∇ loss(x) - ∇ loss(y) |_* ≤ (K_0 + K_ρ * (loss(x) - loss(x*))^ρ) * | x - y |

    where:
    - x, y are current and previous model weights
    - | |_* and | | are dual norms (using Frobenius for now)
    - loss(x*) = 0 (optimal loss value)
    - K_0, ρ, K_ρ are parameters to be fitted
    """

    def __init__(
        self,
        enabled: bool = False,
        max_analysis_steps: int = 1000,
        weight_norm_type: str = 'frobenius',
        rho: float = 2,
    ):
        self.enabled = enabled
        if not self.enabled:
            return

        self.data_points = []  # List of (grad_diff_norm, loss_val, weight_diff_norm)
        self.prev_weights = None
        self.prev_grads = None
        self.max_analysis_steps = max_analysis_steps
        self.rho = rho
        if weight_norm_type == "frobenius":
            self.weight_norm_type = "fro"
            self.grad_norm_type = "fro"
        else:
            raise ValueError(f"Unsupported weight norm type: {weight_norm_type}")
        self.current_step = 0

    def is_enabled(self) -> bool:
        self.enabled = self.enabled and (self.current_step <= self.max_analysis_steps)
        return self.enabled

    def _get_model_weights_flat(self, model: torch.nn.Module) -> torch.Tensor:
        """Get flattened model weights"""
        weights = []
        for param in model.parameters():
            if param.requires_grad:
                weights.append(param.data.view(-1))
        return torch.cat(weights)

    def _get_model_grads_flat(self, model: torch.nn.Module) -> torch.Tensor:
        """Get flattened model gradients"""
        grads = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else None

    def _norm(self, tensor: torch.Tensor, type: str = 'fro') -> float:
        """Compute norm"""
        return torch.norm(tensor, p=type).item()

    def update(self, model: torch.nn.Module, loss_val: float, iteration: int, optimizer_name: str = "unknown"):
        """
        Update analyzer with current model state

        Args:
            model: Current model
            loss_val: Current loss value
            iteration: Current iteration number
            optimizer_name: Name of the optimizer being used
        """
        if not self.is_enabled():
            return

        self.current_step += 1
        # Get current weights and gradients
        current_weights = self._get_model_weights_flat(model)
        current_grads = self._get_model_grads_flat(model)

        if current_grads is None:
            return

        # If we have previous state, compute metrics
        if self.prev_weights is not None and self.prev_grads is not None:
            # Compute | ∇ loss(x) - ∇ loss(y) |_*
            grad_diff = current_grads - self.prev_grads
            grad_diff_norm = self._norm(grad_diff, type=self.grad_norm_type)

            # Compute | x - y |
            weight_diff = current_weights - self.prev_weights
            weight_diff_norm = self._norm(weight_diff, type=self.weight_norm_type)

            # Store data point (assuming loss(x*) = 0)
            if weight_diff_norm > 1e-12:  # Avoid division by zero
                data_point = {
                    'grad_diff_norm': grad_diff_norm,
                    'loss_val': loss_val,
                    'weight_diff_norm': weight_diff_norm,
                    'iteration': iteration
                }
                self.data_points.append(data_point)

                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'lipschitz/grad_diff_norm': grad_diff_norm,
                        'lipschitz/loss_val': loss_val,
                        'lipschitz/weight_diff_norm': weight_diff_norm,
                        'lipschitz/ratio': grad_diff_norm / weight_diff_norm,
                        'iter': iteration
                    })

        # Update previous state
        self.prev_weights = current_weights.clone().detach()
        self.prev_grads = current_grads.clone().detach()

    def _fit_least_squares(
        self, grad_diff_norms: np.ndarray,
        loss_vals: np.ndarray,
        weight_diff_norms: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit K_0, K_ρ using least squares for the linear relationship:
        ratio = K_0 + K_ρ * loss²
        where ratio = |∇loss(x) - ∇loss(y)|_* / |x - y|
        """
        # Compute Lipschitz ratios
        ratios = grad_diff_norms / weight_diff_norms

        # Create design matrix for linear regression: [1, loss**rho]
        loss_squared = loss_vals ** self.rho
        X = np.column_stack([np.ones(len(loss_vals)), loss_squared])
        y = ratios

        # Solve least squares: X * [K_0, K_ρ]ᵀ = y
        try:
            params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            K_0, K_rho = params

            # Ensure positive parameters

            # Calculate R-squared for goodness of fit
            y_pred = K_0 + K_rho * loss_squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'K_0': float(K_0),
                'rho': self.rho,
                'K_rho': float(K_rho),
                'r_squared': float(r_squared),
                'num_data_points': len(ratios)
            }
        except np.linalg.LinAlgError:
            print("Linear algebra error occurred while fitting parameters K_0 and K_rho!")
            return None

    def fit_parameters(self) -> Optional[Dict[str, float]]:
        """
        Fit K_0, K_ρ parameters using least squares with fixed ρ = 2

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

        # Use least squares fitting
        return self._fit_least_squares(grad_diff_norms, loss_vals, weight_diff_norms)

    def plot_results(self, fitted_params: Dict[str, float]) -> Optional[str]:
        """
        Create visualization of the results

        Args:
            fitted_params: Dictionary with fitted K_0, ρ, K_ρ

        Returns:
            Base64 encoded image string or None if failed
        """
        try:
            print(f"Creating plot with {len(self.data_points)} data points...")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract data
            grad_diff_norms = np.array([dp['grad_diff_norm'] for dp in self.data_points])
            loss_vals = np.array([dp['loss_val'] for dp in self.data_points])  # loss(x) - loss(x*) = loss(x) since loss(x*) = 0
            weight_diff_norms = np.array([dp['weight_diff_norm'] for dp in self.data_points])
            iterations = np.array([dp['iteration'] for dp in self.data_points])

            # Main scatter plot - loss²(x) vs ratio with color mapping by iteration
            ratios = grad_diff_norms / weight_diff_norms  # |∇loss(x) - ∇loss(y)|_* / |x - y|
            loss_squared = loss_vals ** self.rho

            # Create color map: blue (early) to red (late iterations)
            scatter = ax.scatter(loss_squared, ratios, c=iterations, cmap='coolwarm',
                               alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Training Iteration', fontsize=10)

            ax.set_xlabel('loss(x)^rho', fontsize=12)
            ax.set_ylabel('||∇loss(x) - ∇loss(y)||_* / ||x - y||', fontsize=12)
            ax.set_title('Lipschitz Analysis: Data vs Fitted Line', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Plot fitted line
            if fitted_params:
                K_0, K_rho = fitted_params['K_0'], fitted_params['K_rho']
                r_squared = fitted_params.get('r_squared', 0)
                loss_sq_range = np.linspace(loss_squared.min(), loss_squared.max(), 100)
                fitted_line = K_0 + K_rho * loss_sq_range  # K_0 + K_ρ * loss²(x)

                ax.plot(loss_sq_range, fitted_line, 'r-', linewidth=3,
                        label=f'Fitted line: K_0 + K_ρ·loss^{self.rho}\nK_0={K_0:.2e}, K_ρ={K_rho:.2e}\nR²={r_squared:.3f}')

                # Add some visual validation
                fitted_vals = K_0 + K_rho * loss_squared
                violations = np.sum(ratios > fitted_vals)
                total_points = len(ratios)
                violation_pct = 100 * violations / total_points
                ax.text(0.02, 0.98, f'Points above line: {violations}/{total_points} ({violation_pct:.1f}%)',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                ax.text(0.02, 0.88, f'Estimated warmup threshold: {(K_0 / ((self.rho - 1) * np.abs(K_rho)))**(1./self.rho):.1f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.legend(fontsize=11)

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

    def finalize_analysis(self):
        """
        Perform final analysis and logging at the end of training
        """

        print("\n=== Lipschitz Analysis Results ===")
        print(f"Collected {len(self.data_points)} data points")

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
            print("Fitted parameters (with fixed ρ = 2):")
            print(f"  K_0 = {fitted_params['K_0']:.6e}")
            print(f"  ρ = {fitted_params['rho']:.1f} (fixed)")
            print(f"  K_ρ = {fitted_params['K_rho']:.6e}")
            print(f"  R² = {fitted_params.get('r_squared', 0):.6f}")

            # Validation check
            grad_norms = [dp['grad_diff_norm'] for dp in self.data_points]
            weight_norms = [dp['weight_diff_norm'] for dp in self.data_points]
            losses = [dp['loss_val'] for dp in self.data_points]

            ratios = np.array(grad_norms) / np.array(weight_norms)
            fitted_vals = fitted_params['K_0'] + fitted_params['K_rho'] * np.array(losses)**self.rho
            violations = np.sum(ratios > fitted_vals)
            print(f"  Validation: {violations}/{len(ratios)} points above bound ({100*violations/len(ratios):.1f}%)")

            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'lipschitz/final_K_0': fitted_params['K_0'],
                    'lipschitz/final_rho': fitted_params['rho'],
                    'lipschitz/final_K_rho': fitted_params['K_rho']
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
                        wandb.log({"lipschitz/analysis_plot": wandb.Image(image)})
                        print("Plot successfully logged to W&B!")
                    except Exception as e:
                        print(f"Error logging plot to W&B: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("Failed to create plot - check matplotlib and PIL installation")
        else:
            print("Failed to fit parameters")
