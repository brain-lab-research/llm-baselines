import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ----------------------------
# 1) Scheduler: eta(Δ) = Δ / (K0 + K1 Δ + K2 Δ^2)
# ----------------------------
def eta_func(x, K0, K1, K2):
    return x / (K0 + K1*x + K2*x**2)

def eta_prime(x, K0, K1, K2):
    D = K0 + K1*x + K2*x**2
    if D <= 0:
        return np.inf
    return (K0 - K2 * x**2) / (D**2)

def eta_double_prime(x, K0, K1, K2):
    D = K0 + K1*x + K2*x**2
    Dp = K1 + 2*K2*x
    N = K0 - K2*x**2
    Np = -2*K2*x
    if D <= 0:
        return np.inf
    return (Np * D - 2 * N * Dp) / (D**3)

# ----------------------------
# 2) Targets
# ----------------------------
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
        return eta_start, 0.0, 0.0
    if x > x_end:
        return eta_end, 0.0, 0.0

    L = x_end - x_start
    cos_arg = np.pi * (x - x_start) / L
    delta_eta = eta_end - eta_start

    val = eta_start + delta_eta * (1 - np.cos(cos_arg)) / 2
    d1 = delta_eta * (np.pi / (2 * L)) * np.sin(cos_arg)
    d2 = delta_eta * (np.pi**2 / (2 * L**2)) * np.cos(cos_arg)

    return val, d1, d2

# ----------------------------
# 3) "LLaMA 124M" shapes (typical small config)
#    If your real 124M differs, edit these constants.
# ----------------------------
def llama124m_shapes(n_layers=12, d_model=768, d_ff=2048, vocab=32000, tied_lm_head=True):
    """
    Returns list of (m,n) for blocks used in max-over-layers norms.
    We include: per layer {Q,K,V,O} (4 * d_model x d_model),
               per layer {up, gate, down} (2 * d_model x d_ff, 1 * d_ff x d_model),
               plus embedding (vocab x d_model),
               plus lm_head if untied.
    """
    shapes = []
    for _ in range(n_layers):
        # attention projections
        shapes += [(d_model, d_model)] * 4  # Q,K,V,O
        # MLP projections (up, gate, down)
        shapes += [(d_model, d_ff)] * 2     # up, gate
        shapes += [(d_ff, d_model)] * 1     # down

    # token embedding
    shapes += [(vocab, d_model)]
    if not tied_lm_head:
        shapes += [(vocab, d_model)]
    return shapes

# ----------------------------
# 4) kappa = sup ||v||_* / ||v|| for max-over-layers norms
# ----------------------------
def kappa_muon(shapes):
    # dual(spectral) = nuclear; ||A||_{S1} <= r ||A||_2 achievable
    return float(sum(min(m, n) for (m, n) in shapes))

def kappa_sign(shapes):
    # dual(inf) = 1; ||A||_1 <= d ||A||_inf achievable
    return float(sum(m * n for (m, n) in shapes))

def kappa_norm(shapes):
    # dual(F) = F; sup_{max ||v_i||_F=1} sum ||v_i||_F = L
    return float(len(shapes))

# ----------------------------
# 5) Gaussian weight around Δ*
#    w(Δ;Δ*) = exp(-(Δ-Δ*)^2 / (2*sigma_norm)), with sigma_F=1.
#    make_w_delta depends ONLY on one number: sigma_norm.
# ----------------------------
def make_w_delta(sigma_norm, sigma_F=1.0):
    sigma = float(sigma_norm)
    if sigma <= 0:
        raise ValueError("sigma_norm must be > 0")

    inv_2sigma = sigma_F / (2.0 * sigma)

    def w(delta, delta_star):
        d = delta - delta_star
        return np.exp(-(d * d) * inv_2sigma)

    return w

# ----------------------------
# 6) Main solver
# ----------------------------
def compute_lipschitz_constants(
    lr, lr_0, delta0,
    target='cos', mode="linear_and_cos",
    w_delta=None,
    verbose=False,
    loss_star=None,
    loss0=None
):
    div = lr / lr_0
    if delta0 <= 0:
        raise ValueError("delta0 must be > 0")

    def objective(params, delta_star):
        K0, K1, K2 = params

        # Hard constraints
        if abs(K0 - K2 * delta_star**2) > 1e-10:
            return np.inf
        if abs(eta_func(delta_star, K0, K1, K2) - lr) > 1e-10:
            return np.inf
        if abs(eta_func(delta0, K0, K1, K2) - lr/div) > 1e-10:
            return np.inf

        x_start = 0.0 if mode in ["linear_and_cos", "max_linear_and_cos", "linear_and_linaer",
                                  "intergral_0_x0", "func_prime_0x*"] else delta_star
        x_end = delta_star if mode in ["func_prime_0x*"] else delta0

        eta_x_start = eta_func(x_start, K0, K1, K2)
        eta_x_end = eta_func(x_end, K0, K1, K2)

        # target function for "single-interval" modes
        if target == 'linear':
            def target_func(x):
                return linear_target(x, x_start, x_end, eta_x_start, eta_x_end)
        elif target == 'cos':
            def target_func(x):
                return cosine_target(x, x_start, x_end, eta_x_start, eta_x_end)
        else:
            raise ValueError(f"Unknown target='{target}'")

        def integrand(x):
            w = w_delta(x, delta_star)

            if mode in ["func_prime", "func_prime_0x*"]:
                eta_val = eta_func(x, K0, K1, K2)
                eta_der = eta_prime(x, K0, K1, K2)
                target_val, target_der, _ = target_func(x)
                return w * (abs(eta_val - target_val) + abs(eta_der - target_der))

            elif mode == "double_prime":
                _, _, target_d2 = target_func(x)
                return w * abs(eta_double_prime(x, K0, K1, K2) - target_d2)

            elif mode in ["linear_and_cos", "max_linear_and_cos"]:
                eta_val = eta_func(x, K0, K1, K2)
                if x < delta_star:
                    target_val, _, _ = cosine_target(
                        x, 0.0, delta_star,
                        eta_func(0.0, K0, K1, K2),
                        eta_func(delta_star, K0, K1, K2),
                    )
                else:
                    target_val, _, _ = linear_target(
                        x, delta_star, delta0,
                        eta_func(delta_star, K0, K1, K2),
                        eta_func(delta0, K0, K1, K2),
                    )
                return w * (abs(eta_val - target_val) ** 2)

            elif mode == "linear_and_linaer":
                eta_val = eta_func(x, K0, K1, K2)
                if x < delta_star:
                    target_val, _, _ = linear_target(
                        x, 0.0, delta_star,
                        eta_func(0.0, K0, K1, K2),
                        eta_func(delta_star, K0, K1, K2),
                    )
                else:
                    target_val, _, _ = linear_target(
                        x, delta_star, delta0,
                        eta_func(delta_star, K0, K1, K2),
                        eta_func(delta0, K0, K1, K2),
                    )
                return w * abs(eta_val - target_val)

            elif mode == "intergral_0_x0":
                # maximize area under eta => minimize negative area (weighted)
                return -w * eta_func(x, K0, K1, K2)

            else:
                raise ValueError(f"Unknown mode='{mode}'")

        if mode == "max_linear_and_cos":
            integral1, _ = quad(integrand, x_start, delta_star)
            integral2, _ = quad(integrand, delta_star, x_end)
            denom1 = max(abs(x_start - delta_star), 1e-12)
            denom2 = max(abs(x_end - delta_star), 1e-12)
            return integral1 / denom1 + integral2 / denom2
        else:
            integral, _ = quad(integrand, x_start, x_end)
            denom = max(abs(x_end - x_start), 1e-12)
            return integral / denom

    def K_from_delta_star(delta_star):
        K2 = delta0 * (div - 1) / (lr * (delta0 - delta_star) ** 2)
        K0 = K2 * delta_star ** 2
        K1 = 1 / lr - 2 * K2 * delta_star
        return K0, K1, K2

    # scan delta_star
    delta_star_values = np.linspace(0.01, delta0 - 0.01, 500)
    best = (np.inf, None, None)

    for ds in delta_star_values:
        K0, K1, K2 = K_from_delta_star(ds)
        val = objective((K0, K1, K2), ds)
        if val < best[0]:
            best = (val, (K0, K1, K2), ds)

    min_obj, (K0, K1, K2), delta_star = best

    print(f"Δ0 = {delta0:.6f}")
    print(f"Optimal Δ* = {delta_star:.6f}")
    if (loss_star is not None) and (loss0 is not None):
        print(f"loss* = {loss_star:.6f}, loss0 = {loss0:.6f}, implied loss(Δ*) = {loss_star + delta_star:.6f}")
    print(f"K0 = {K0:.10f}")
    print(f"K1 = {K1:.10f}")
    print(f"K2 = {K2:.10f}")
    print(f"Objective = {min_obj:.10e}")
    print("Check:")
    print(f"eta(Δ*) = {eta_func(delta_star, K0, K1, K2):.6e}  (target lr={lr:.6e})")
    print(f"eta(Δ0) = {eta_func(delta0, K0, K1, K2):.6e}  (target lr/div={lr/div:.6e})")

    if verbose:
        X = np.linspace(0, delta0, 800)
        eta_vals = np.array([eta_func(x, K0, K1, K2) for x in X])

        # build piecewise target for plotting (matches mode linear_and_cos)
        tgt = np.zeros_like(X)
        for j, x in enumerate(X):
            if mode in ["linear_and_cos", "max_linear_and_cos"]:
                if x < delta_star:
                    tgt[j] = cosine_target(
                        x, 0.0, delta_star,
                        eta_func(0.0, K0, K1, K2),
                        eta_func(delta_star, K0, K1, K2),
                    )[0]
                else:
                    tgt[j] = linear_target(
                        x, delta_star, delta0,
                        eta_func(delta_star, K0, K1, K2),
                        eta_func(delta0, K0, K1, K2),
                    )[0]
            else:
                x_start = 0.0
                x_end = delta0
                tgt[j] = cosine_target(
                    x, x_start, x_end,
                    eta_func(x_start, K0, K1, K2),
                    eta_func(x_end, K0, K1, K2)
                )[0]

        plt.figure(figsize=(9, 5))
        plt.plot(X, eta_vals, lw=2, label=r'$\eta(\Delta)$')
        plt.plot(X, tgt, lw=2, ls='--', label='target')
        plt.axvline(delta0, color='red', ls='--', label=r'$\Delta_0$')
        plt.axvline(delta_star, color='green', ls=':', label=r'$\Delta_*$')
        plt.scatter([delta_star], [eta_func(delta_star, K0, K1, K2)], s=60)
        plt.grid(True)
        plt.xlabel(r'$\Delta$')
        plt.ylabel(r'$\eta(\Delta)$')
        plt.title(f"mode={mode}, target={target}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if w_delta is not None:
            # visualize Gaussian weight centered at the optimal Δ*
            W = np.array([w_delta(x, delta_star) for x in X])
            plt.figure(figsize=(9, 4))
            plt.plot(X, W, lw=2)
            plt.grid(True)
            plt.xlabel(r'$\Delta$')
            plt.ylabel(r'$w(\Delta;\Delta_*)$')
            plt.title("Gaussian weight function")
            plt.tight_layout()
            plt.show()

    return K0, K1, K2, delta_star

if __name__ == "__main__":
    # ----------------------------
    # 7) Example usage: choose optimizer norm, build sigma_norm from ||·||_*/||·||, run
    # ----------------------------
    lr = 1e-3
    div = 100
    loss0 = 11.0
    loss_star = 3.2
    delta0 = loss0 - loss_star

    # choose which norm you want:
    # "muon", "sign", "norm"
    which = "sign"

    shapes = llama124m_shapes(
        n_layers=12, d_model=768, d_ff=2048, vocab=32000, tied_lm_head=True
    )

    # sigma_norm depends ONLY on kappa = sup ||v||_* / ||v||
    if which == "muon":
        kappa = kappa_muon(shapes)
    elif which == "sign":
        kappa = kappa_sign(shapes)
    elif which == "norm":
        kappa = kappa_norm(shapes)
    else:
        raise ValueError("which must be in {'muon','sign','norm'}")

    sigma_norm = 1 / kappa
    sigma_F = 0.001

    w_delta = make_w_delta(sigma_norm, sigma_F=sigma_F)

    print(f"[{which}] kappa={kappa:.6e}, sigma_norm={sigma_norm:.6e}, sigma_F={sigma_F:.6e}")

    K0, K1, K2, delta_star = compute_lipschitz_constants(
        lr=lr,
        lr_0=lr/div,
        delta0=delta0,
        target="cos",
        mode="linear_and_cos",
        w_delta=w_delta,
        verbose=True,
        loss_star=loss_star,
        loss0=loss0,
    )