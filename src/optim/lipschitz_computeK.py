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

def linear_target(x, x_star, x0, eta_x_star, eta_x0):
    slope = (eta_x0 - eta_x_star) / (x0 - x_star)
    val = eta_x_star + slope * (x - x_star)
    deriv = slope
    return val, deriv

def cosine_target(x, x_star, x0, eta_x_star, eta_x0):
    # Cosine interpolation between eta_x_star and eta_x0 on [x_star, x0]
    if x < x_star or x > x0:
        # Outside interval return boundary values
        if x < x_star:
            return eta_x_star, 0.0
        else:
            return eta_x0, 0.0
    L = x0 - x_star
    cos_arg = np.pi * (x - x_star) / L
    val = eta_x_star + (eta_x0 - eta_x_star) * (1 - np.cos(cos_arg)) / 2
    deriv = (eta_x0 - eta_x_star) * (np.pi / (2*L)) * np.sin(cos_arg)
    return val, deriv

# Main function

def compute_lipschitz_constants(lr, lr_0, x0, target='linear', verbose=False):
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

        # Select target function and derivative
        if target == 'linear':
            def target_func(x):
                val, deriv = linear_target(x, x_star, x0, eta_x_star, eta_x0)
                return val, deriv
        elif target == 'cosine':
            def target_func(x):
                val, deriv = cosine_target(x, x_star, x0, eta_x_star, eta_x0)
                return val, deriv
        else:
            raise ValueError(f"Unknown target function '{target}'")

        def integrand(x):
            eta_val = eta_func(x, K0, K1, K2)
            eta_der = eta_prime(x, K0, K1, K2)
            target_val, target_der = target_func(x)
            return abs(eta_val - target_val) + abs(eta_der - target_der)

        try:
            integral, _ = quad(integrand, x_star, x0)
        except:
            return np.inf

        return abs(integral) / max(abs(x0 - x_star), 1e-12)

    def optimize_x_star(x_star):
        if x_star >= x0 or x_star <= 0:
            return np.inf, 0, 0, 0
        try:
            K2 = x0*(div - 1)/(lr*(x0 - x_star)**2)
            K0 = K2 * x_star**2
            K1 = 1/lr - 2*K2*x_star
        except:
            return np.inf, 0, 0, 0

        params = [K0, K1, K2]
        obj_value = objective(params, x_star, x0, lr, div)
        return obj_value, K0, K1, K2

    x_star_values = np.linspace(0.1, x0 - 0.1, 50)
    obj_values = []
    K_params = []

    for xs in x_star_values:
        try:
            obj_val, K0, K1, K2 = optimize_x_star(xs)
            obj_values.append(obj_val)
            K_params.append((K0, K1, K2))
        except:
            obj_values.append(np.inf)
            K_params.append((0, 0, 0))

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
        else:  # cosine
            target_vals = [cosine_target(x, optimal_x_star, x0, eta_x_star, eta_x0)[0] for x in X if optimal_x_star <= x <= x0]
            target_x = [x for x in X if optimal_x_star <= x <= x0]

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
