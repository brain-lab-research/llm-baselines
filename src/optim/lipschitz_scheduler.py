"""
Lipschitz-based Learning Rate Scheduler

This scheduler computes the learning rate based on the current loss and
Lipschitz constant estimates:

    lr_t = Δ_t / (K_0 + K_1 * Δ_t + K_rho * Δ_t^ρ)
    K_1 = 0 => lr_t = Δ_t / (K_0 + K_rho * Δ_t^ρ)
    lt_t_max = lr. Delta_t^* (rho = 2) = sqrt(K_0 / K_rho) => lt_t_max = sqrt(K_0 / K_rho) / (K_0 + K_0) = sqrt(1 / (4 K_0 K_rho)) = lr
    => K_0 = 1 / (4 lr^2 K_rho) ~ 1 / (C * lr^2 K_rho)

where Δ_t = loss_t - loss_star (the optimality gap).
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import wandb

class LipschitzScheduler(_LRScheduler):
    """
    Lipschitz-based learning rate scheduler.

    The learning rate is computed as:
        lr_t = Δ_t / (K_0 + K_1 * Δ_t + K_rho * Δ_t^ρ)

    where:
        - Δ_t = max(loss_t - loss_star, epsilon) is the optimality gap
        - K_0, K_1, K_rho are Lipschitz constant estimates
        - ρ (rho) is the power parameter (typically > 1)

    Args:
        optimizer: Wrapped optimizer
        K_0: Constant term in denominator (default: 1.0)
        K_1: Linear coefficient (default: 0.1)
        K_rho: Power term coefficient (default: 0.01)
        rho: Power parameter (default: 2.0)
        loss_star: Estimated optimal loss value (default: 0.0)
        min_lr: Minimum learning rate (default: 1e-6)
        max_lr: Maximum learning rate (default: 1.0)
        epsilon: Small constant to avoid division by zero (default: 1e-8)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer,
        K_0=None,
        K_1=0,
        K_rho=None,
        rho=2.0,
        loss_star=0.0,
        min_lr=1e-6,
        max_lr=1.0,
        epsilon=1e-8,
        last_epoch=-1,
        adjust_K=False,
        target="linear",
        lr=None,
    ):
        if not adjust_K:
            self.K_0 = K_0
            self.K_1 = K_1
            self.K_rho = K_rho
            print(f"Using parameters via args:\nK_0={self.K_0}, K_1={self.K_1}, K_rho={self.K_rho}")
        self.rho = rho
        self.loss_star = loss_star
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epsilon = epsilon
        self.target = target
        self.adjust_K = adjust_K

        # Current loss value (updated via step(loss))
        self.current_loss = None

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        """
        Compute learning rate for each parameter group.
        """
        # If no loss provided yet, return base learning rates
        if self.current_loss is None:
            print("No loss provided for Lipschitz Scheduler!")
            return self.base_lrs

        # Compute optimality gap
        delta_t = max(self.current_loss - self.loss_star, self.epsilon)

        # Compute denominator: K_0 + K_1 * Δ_t + K_rho * Δ_t^ρ
        denominator = self.K_0 + self.K_1 * delta_t + self.K_rho * (delta_t ** self.rho)

        # Avoid division by zero
        denominator = max(denominator, self.epsilon)

        # Compute learning rate: lr_t = Δ_t / denominator
        lr = delta_t / denominator

        # Clamp to [min_lr, max_lr]
        lr = max(self.min_lr, min(lr, self.max_lr))

        # Scale proportionally to base learning rates for each param group
        # This allows different param groups to have different relative LRs
        if len(self.base_lrs) == 1:
            return [lr]
        else:
            # Scale according to the ratio of base_lrs
            base_lr_sum = sum(self.base_lrs)
            if base_lr_sum > 0:
                return [lr * (base_lr / self.base_lrs[0]) for base_lr in self.base_lrs]
            else:
                return [lr] * len(self.base_lrs)

    def step(self, loss=None, epoch=None):
        """
        Update learning rate.

        Args:
            loss: Current loss value (required for computing lr_t)
            epoch: Manual epoch number (optional)
        """
        if self.adjust_K and self.current_loss is None and loss is not None:
            assert self.rho == 2, "rho must equal to 2 for this weird formulas"
            # lr, lr_0 = self.max_lr, self.min_lr
            # div = lr / lr_0
            # x0 = loss - self.loss_star
            # sqrt_term = math.sqrt(4*div + 1)
            # self.K_rho = 4 * div**2 * (div - 1) / (lr * x0 * (sqrt_term - 1)**2)
            # self.K_0 = x0 * (div - 1) * (2*div + 1 - sqrt_term)**2 / (lr * (sqrt_term - 1)**2)
            # self.K_1 = 1/lr - 4 * div * (div - 1) * (2*div + 1 - sqrt_term) / (lr * (sqrt_term - 1)**2)

            # print(f"Using parameters via lr and min_lr:\nK_0={self.K_0}, K_1={self.K_1}, K_rho={self.K_rho}")
            # x_star = x0 * (2*div + 1 - sqrt_term) / (2*div)
            # print(f"Estimated Delta_t threshold: Delta_t = loss(x_t) - loss* = loss(x_t) - {self.loss_star} = {x_star}")

            from .lipschitz_computeK import compute_lipschitz_constants
            self.K_0, self.K_1, self.K_rho, delta_star = compute_lipschitz_constants(
                lr=self.max_lr, lr_0=self.min_lr, x0=loss - self.loss_star, target=self.target)
            if wandb.run:
                wandb.log({
                    "lipschitz_K_0": self.K_0,
                    "lipschitz_K_1": self.K_1,
                    "lipschitz_K_rho": self.K_rho,
                    "lipschitz_delta_star": delta_star
                })

        if loss is not None:
            self.current_loss = float(loss)

        # Call parent step to update last_epoch and apply new lr
        super().step(epoch)

    def state_dict(self):
        """
        Return the state of the scheduler as a dict.
        """
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ('optimizer', 'is_better')
        }
        return state

    def load_state_dict(self, state_dict):
        """
        Load the scheduler's state.
        """
        self.__dict__.update(state_dict)
