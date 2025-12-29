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
        decay_scheduler: Scheduler to use after Lipschitz phase
        decay_scheduler_args: Args object for recreating decay_scheduler
        decay_scheduler_group_specs: Group specs for recreating decay_scheduler
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
        max_steps=-1,
        mode="func_prime",
        decay_scheduler=None,
        decay_scheduler_args=None,
        decay_scheduler_group_specs=None,
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
        self.adjust_K = adjust_K

        self.current_loss = None

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.max_steps = max_steps
        self.mode = mode
        self.target = target
        self.decay_scheduler = decay_scheduler
        self.decay_scheduler_args = decay_scheduler_args
        self.decay_scheduler_group_specs = decay_scheduler_group_specs
        self.current_step = 0
        self.use_decay_scheduler = False

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.current_loss is None:
            print("No loss provided for Lipschitz Scheduler!")
            return self.base_lrs
        
        self.current_step += 1

        if self.use_decay_scheduler and self.decay_scheduler is not None:
            return self.decay_scheduler.get_last_lr()

        delta_t = max(self.current_loss - self.loss_star, self.epsilon)
        if delta_t <= self.delta_star and self.decay_scheduler is not None:
            # Re-initialize decay_scheduler with correct total_steps
            if not self.use_decay_scheduler:
                self._reinit_decay_scheduler()
            return self.decay_scheduler.get_last_lr()

        denominator = self.K_0 + self.K_1 * delta_t + self.K_rho * (delta_t ** self.rho)
        denominator = max(denominator, self.epsilon)

        lr = delta_t / denominator
        lr = max(0, min(lr, self.max_lr))

        if len(self.base_lrs) == 1:
            return [lr]
        else:
            base_lr_sum = sum(self.base_lrs)
            if base_lr_sum > 0:
                return [lr * (base_lr / self.base_lrs[0]) for base_lr in self.base_lrs]
            else:
                return [lr] * len(self.base_lrs)

    def _reinit_decay_scheduler(self):
        """Re-initialize decay_scheduler with correct parameters using get_scheduler."""
        if self.decay_scheduler is None:
            return

        remaining_steps = self.max_steps - self.current_step + 1
        print(f"Re-initializing decay_scheduler with remaining_steps={remaining_steps}")

        if self.decay_scheduler_args is not None:
            try:
                from .schedule import get_scheduler
                self.decay_scheduler = get_scheduler(
                    self.optimizer,
                    self.decay_scheduler_args,
                    n_iterations=remaining_steps,
                    group_specs=self.decay_scheduler_group_specs
                )
                print(f"Successfully re-initialized decay_scheduler using get_scheduler")
                self.use_decay_scheduler = True
            except Exception as e:
                print(f"Warning: Failed to re-initialize decay_scheduler: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: decay_scheduler_args not provided, cannot re-initialize scheduler")

    def step(self, loss=None, epoch=None):
        if self.adjust_K and self.current_loss is None and loss is not None:
            from .lipschitz_computeK import compute_lipschitz_constants
            self.K_0, self.K_1, self.K_rho, self.delta_star = compute_lipschitz_constants(
                lr=self.max_lr,
                lr_0=self.min_lr,
                x0=loss - self.loss_star,
                target=self.target,
                mode=self.mode,
            )
            if wandb.run:
                print("Logging Lipschitz constants to wandb")
                wandb.log({
                    "lipschitz_K_0": self.K_0,
                    "lipschitz_K_1": self.K_1,
                    "lipschitz_K_rho": self.K_rho,
                    "lipschitz_delta_star": self.delta_star
                })

        if loss is not None:
            self.current_loss = float(loss)
        super().step(epoch)

        if self.use_decay_scheduler and self.decay_scheduler is not None:
            self.decay_scheduler.step()
        

    def state_dict(self):
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ('optimizer', 'is_better', 'decay_scheduler', 'decay_scheduler_args', 'decay_scheduler_group_specs')
        }
        if self.decay_scheduler is not None:
            state['decay_scheduler_state'] = self.decay_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        decay_scheduler_state = state_dict.pop('decay_scheduler_state', None)
        self.__dict__.update(state_dict)
        if decay_scheduler_state is not None and self.decay_scheduler is not None:
            self.decay_scheduler.load_state_dict(decay_scheduler_state)
