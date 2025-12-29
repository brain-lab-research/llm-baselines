import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Eta function
def eta_func(x, K0, K1, K2):
    return x / (K0 + K1*x + K2*x**2)

# First derivative of eta
def eta_prime(x, K0, K1, K2):
    D = K0 + K1*x + K2*x**2
    if D <= 0:
        return np.inf
    return (K0 - K2 * x**2) / (D**2)

# Second derivative of eta
def eta_double_prime(x, K0, K1, K2):
    D = K0 + K1*x + K2*x**2
    Dp = K1 + 2*K2*x
    N = K0 - K2*x**2
    Np = -2*K2*x
    if D <= 0:
        return np.inf
    return (Np * D - 2 * N * Dp) / (D**3)

# Target functions and their derivatives

def linear_target(x, x_start, x_end, eta_start, eta_end):
    slope = (eta_end - eta_start) / (x_end - x_start)
    val = eta_start + slope * (x - x_start)
    deriv = slope
    return val, deriv, 0.0

def cosine_target(x, x_start, x_end, eta_start, eta_end):
    if x_start > x_end:
        x_start, x_end = x_end, x_start
        eta_start, eta_end = eta_end, eta_start

    if x < x_start:
        print("!!!!")
        return eta_start, 0.0, 0.0
    if x > x_end:
        print("????")
        return eta_end, 0.0, 0.0

    L = x_end - x_start
    cos_arg = np.pi * (x - x_start) / L
    delta_eta = eta_end - eta_start

    # f(x)
    val = eta_start + delta_eta * (1 - np.cos(cos_arg)) / 2

    # f'(x)
    d1 = delta_eta * (np.pi / (2 * L)) * np.sin(cos_arg)

    # f''(x)
    d2 = delta_eta * (np.pi**2 / (2 * L**2)) * np.cos(cos_arg)

    return val, d1, d2

# Main function

def compute_lipschitz_constants(lr, lr_0, x0, target='linear', mode="func_prime", verbose=False):
    """
    Computes coefficients K0, K1, K2 for learning rate scheduler:
        eta(x) = x / (K0 + K1*x + K2*x^2)

    Parameters:
        lr (float): max learning rate value at x_star
        lr_0 (float): min learning rate value at x0
        x0 (float): starting loss(x0) - loss*
        target (str): target warmup function type ('linear' or 'cosine')
        verbose (bool): print info and plot graph if True

    Returns:
        tuple: (K0, K1, K2, delta_star)
    """

    div = lr / lr_0

    def objective(params, x_star, x0, lr, div):
        K0, K1, K2 = params

        # Constraint 1: eta'(x_star) = 0 => K0 = K2 * x_star^2
        if abs(K0 - K2 * x_star**2) > 1e-10:
            return np.inf

        # Constraint 2: eta(x_star) = lr
        if abs(eta_func(x_star, K0, K1, K2) - lr) > 1e-10:
            return np.inf

        # Constraint 3: eta(x0) = lr/div
        if abs(eta_func(x0, K0, K1, K2) - lr/div) > 1e-10:
            return np.inf

        eta_x_star = eta_func(x_star, K0, K1, K2)
        eta_x0 = eta_func(x0, K0, K1, K2)
        if abs(x0 - x_star) < 1e-12:
            return np.inf
        
        x_start = 0.0 if mode in ["intergral_0_x0", "func_prime_0x*"] else x_star
        x_end = x_star if mode in ["func_prime_0x*"] else x0

        # Select target function and derivative
        if target == 'linear':
            def target_func(x):
                val, deriv, d2 = linear_target(x, x_start, x_end, eta_x_star, eta_x0)
                return val, deriv, d2
        elif target == 'cos':
            def target_func(x):
                val, deriv, d2 = cosine_target(x, x_start, x_end, eta_x_star, eta_x0)
                return val, deriv, d2
        else:
            raise ValueError(f"Unknown target function '{target}'")

        def integrand(x):
            if mode in ["func_prime", "func_prime_0x*"]:
                eta_val = eta_func(x, K0, K1, K2)
                eta_der = eta_prime(x, K0, K1, K2)
                target_val, target_der, _ = target_func(x)
                return abs(eta_val - target_val) # + abs(eta_der - target_der)
            elif mode == "double_prime":
                _, _, d2 = target_func(x)
                return abs(eta_double_prime(x, K0, K1, K2) - d2)
            elif mode == "intergral_0_x0":
                return -eta_func(x, K0, K1, K2)
        
        # try:
        integral, _ = quad(integrand, x_start, x_end)
        # except:
        #     return np.inf

        return integral / max(abs(x_end - x_start), 1e-12)

    def optimize_x_star(x_star):
        if x_star >= x0 or x_star <= 0:
            return np.inf, 0, 0, 0
        # try:
        K2 = x0*(div - 1)/(lr*(x0 - x_star)**2)
        K0 = K2 * x_star**2
        K1 = 1/lr - 2*K2*x_star
        # except:
        #     return np.inf, 0, 0, 0

        params = [K0, K1, K2]
        obj_value = objective(params, x_star, x0, lr, div)
        return obj_value, K0, K1, K2

    x_star_values = np.linspace(0.1, x0 - 0.1, 50)
    obj_values = []
    K_params = []

    for xs in x_star_values:
        # try:
        obj_val, K0, K1, K2 = optimize_x_star(xs)
        obj_values.append(obj_val)
        K_params.append((K0, K1, K2))
        # except:
        #     obj_values.append(np.inf)
        #     K_params.append((0, 0, 0))

    valid_indices = [i for i, val in enumerate(obj_values) if not np.isinf(val)]
    if len(valid_indices) == 0:
        raise ValueError("No valid solution found.")

    min_idx = valid_indices[np.argmin([obj_values[i] for i in valid_indices])]
    optimal_x_star = x_star_values[min_idx]
    optimal_K0, optimal_K1, optimal_K2 = K_params[min_idx]
    min_objective = obj_values[min_idx]

    print(f"Starting x0 = {x0:.4f}")
    print(f"Optimal x* = {optimal_x_star:.4f}")
    print(f"K0 = {optimal_K0:.6f}")
    print(f"K1 = {optimal_K1:.6f}")
    print(f"K2 = {optimal_K2:.6f}")
    print(f"Minimum objective value = {min_objective:.8f}")

    print("\nVerification:")
    print(f"eta(x*) = {eta_func(optimal_x_star, optimal_K0, optimal_K1, optimal_K2):.6f} (should be {lr})")
    print(f"eta(x0) = {eta_func(x0, optimal_K0, optimal_K1, optimal_K2):.6f} (should be {lr/div})")

    if verbose:
        X = np.linspace(0, x0, 500)
        Y = [eta_func(x, optimal_K0, optimal_K1, optimal_K2) for x in X]

        # Prepare target function values for plotting
        eta_x_star = eta_func(optimal_x_star, optimal_K0, optimal_K1, optimal_K2)
        eta_x0 = eta_func(x0, optimal_K0, optimal_K1, optimal_K2)
        if target == 'linear':
            target_vals = [linear_target(x, optimal_x_star, x0, eta_x_star, eta_x0)[0] for x in X if optimal_x_star <= x <= x0]
            target_x = [x for x in X if optimal_x_star <= x <= x0]
        elif target == 'cos':
            target_vals = [cosine_target(x, optimal_x_star, x0, eta_x_star, eta_x0)[0] for x in X if optimal_x_star <= x <= x0]
            target_x = [x for x in X if optimal_x_star <= x <= x0]
        else:
            raise ValueError(f"Unknown target function '{target}'")

        plt.figure(figsize=(8,5))
        plt.plot(X, Y, label=r'$\eta(\Delta_t)$', color='blue', lw=2)
        plt.plot(target_x, target_vals, color='gray', lw=2, linestyle='--', label=f'Target warmup({target})')
        plt.axvline(x0, color='red', linestyle='--', label=rf'$\Delta_0={x0}$')
        plt.axvline(optimal_x_star, color='green', linestyle=':', label=rf'$\Delta_*={optimal_x_star:.2f}$')
        plt.scatter(optimal_x_star, eta_func(optimal_x_star, optimal_K0, optimal_K1, optimal_K2),
                   color='green', s=80, zorder=5,
                   label=rf'$\eta(\Delta_*)=${eta_func(optimal_x_star, optimal_K0, optimal_K1, optimal_K2):.4f}')
        plt.scatter(x0, eta_func(x0, optimal_K0, optimal_K1, optimal_K2),
                   color='red', s=80, zorder=5,
                   label=rf'$\eta(\Delta_0)=${eta_func(x0, optimal_K0, optimal_K1, optimal_K2):.4f}')
        plt.xlabel(r'$\Delta_t$ (residual)')
        plt.ylabel(r'$\eta(\Delta_t)$ (learning rate step)')
        plt.title(f'Optimal $\eta(\Delta_t)$ for lr={lr}, div={div}, \Delta_0={x0}, target warmup={target}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return optimal_K0, optimal_K1, optimal_K2, optimal_x_star

if __name__ == "__main__":
    # Example usage
    lr = 1e-3
    div = 100
    loss_0 = 11
    loss_star = 3.45
    target = 'cos'  # 'linear' or 'cos'
    mode = "func_prime_0x*"  # "func_prime", "double_prime", or "intergral_0_x0"

    compute_lipschitz_constants(
        lr, lr / div, loss_0 - loss_star, 
        target=target, mode=mode, verbose=True
    )
