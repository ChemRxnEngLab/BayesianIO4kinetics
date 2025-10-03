import typing as t

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor as pt

from data.rastrigin import rastrigin, noisy_rastrigin, params


def gen_data(n: int = 100, x_range: float = 5.0, y_const: t.Optional[float] = None):
    """Generate n 2D sample locations and noisy observations of the rastrigin function.

    Returns X1, X2, Y arrays.
    """
    X = np.random.uniform(-x_range, x_range, size=(n, 2))
    if y_const is not None:
        X[:, 1] = y_const
    return X[:, 0].copy(), X[:, 1].copy(), noisy_rastrigin(X, params)


def nonp_rastrigin(X1: np.ndarray, X2: np.ndarray, params_list: t.List[float]):
    """Evaluate the separable Rastrigin-style function component-wise.

    This is a NumPy implementation used for predictions and plotting.
    """
    A0 = params_list[0]
    A = params_list[1:3]
    omega = params_list[3:5]
    chi = params_list[5:7]
    val = A0
    val = val + (X1 - chi[0]) ** 2 - A[0] * np.cos(omega[0] * (X1 - chi[0]))
    val = val + (X2 - chi[1]) ** 2 - A[1] * np.cos(omega[1] * (X2 - chi[1]))
    return val


def pt_rastrigin(X1, X2, params_list):
    """Evaluate the separable Rastrigin-style function component-wise using PyTensor ops.

    This is a PyTensor implementation used inside the PyMC model.
    """
    A0 = params_list[0]
    A = params_list[1:3]
    omega = params_list[3:5]
    chi = params_list[5:7]
    val = A0
    val = val + (X1 - chi[0]) ** 2 - A[0] * pt.math.cos(omega[0] * (X1 - chi[0]))
    val = val + (X2 - chi[1]) ** 2 - A[1] * pt.math.cos(omega[1] * (X2 - chi[1]))
    return val


def build_model(
    X1: np.ndarray,
    X2: np.ndarray,
    Y: np.ndarray,
    prior_overrides: t.Optional[t.Dict[str, t.Tuple[float, float]]] = None,
) -> pm.Model:
    """Constructs a PyMC model for the Rastrigin parameters given data.

    Note: this model currently uses NumPy arrays for X1, X2, Y in the likelihood.
    Converting to `pm.Data` and `pm.math` makes it fully PyTensor-compatible.
    """
    # prior_overrides: dict of name -> (mu, sigma) to use Normal priors centered at mu
    with pm.Model() as model:
        if prior_overrides is None:
            A0 = pm.HalfNormal("A0", sigma=10)
            A1 = pm.HalfNormal("A1", sigma=10)
            A2 = pm.HalfNormal("A2", sigma=10)
            omega1 = pm.Uniform("omega1", lower=1, upper=4)
            omega2 = pm.Uniform("omega2", lower=1, upper=4)
            chi1 = pm.Uniform("chi1", lower=0, upper=2 * np.pi)
            chi2 = pm.Uniform("chi2", lower=0, upper=2 * np.pi)
        else:

            def _norm(name):
                mu_val, sigma_val = prior_overrides.get(name, (0.0, 1.0))
                sigma_val = float(max(sigma_val, 1e-3))
                return pm.Normal(name, mu=float(mu_val), sigma=float(sigma_val))

            A0 = _norm("A0")
            A1 = _norm("A1")
            A2 = _norm("A2")
            omega1 = _norm("omega1")
            omega2 = _norm("omega2")
            chi1 = _norm("chi1")
            chi2 = _norm("chi2")

        mu = nonp_rastrigin(X1, X2, [A0, A1, A2, omega1, omega2, chi1, chi2])

        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    return model


def sample_posterior(
    model: pm.Model, draws: int = 1000, tune: int = 500, **sample_kwargs
) -> az.InferenceData:
    """Sample from the posterior and return an ArviZ InferenceData object."""
    with model:
        trace = pm.sample(draws, tune=tune, return_inferencedata=True, **sample_kwargs)
    return trace


def posterior_param_samples(trace: az.InferenceData) -> t.Dict[str, np.ndarray]:
    """Extract posterior samples for parameters and stack chain/draw into samples axis.

    Returns a dict mapping parameter name -> (n_samples,) or (n_samples, dim) array.
    """
    posterior = trace.posterior

    def stack(arr):
        return arr.values.reshape(-1, *arr.shape[2:])

    out = {
        "A0": stack(posterior["A0"]),
        "A1": stack(posterior["A1"]),
        "A2": stack(posterior["A2"]),
        "omega1": stack(posterior["omega1"]),
        "omega2": stack(posterior["omega2"]),
        "chi1": stack(posterior["chi1"]),
        "chi2": stack(posterior["chi2"]),
    }
    return out


def predictive_samples_on_grid(
    X_grid: np.ndarray, param_samples: t.Dict[str, np.ndarray], max_samps: int = 500
) -> np.ndarray:
    """Compute predictive samples at each point in a 2D grid.

    X_grid should be shape (n_points, 2).
    Returns array shape (n_use, n_points).
    """
    n_samps = param_samples["A0"].shape[0]
    n_use = min(max_samps, n_samps)
    idx = np.linspace(0, n_samps - 1, n_use).astype(int)

    preds = np.zeros((n_use, X_grid.shape[0]))
    for i, ii in enumerate(idx):
        ps = [
            float(param_samples["A0"][ii]),
            float(param_samples["A1"][ii]),
            float(param_samples["A2"][ii]),
            float(param_samples["omega1"][ii]),
            float(param_samples["omega2"][ii]),
            float(param_samples["chi1"][ii]),
            float(param_samples["chi2"][ii]),
        ]
        preds[i, :] = rastrigin(X_grid, ps)
    return preds


def acquisition_lcb(
    pred_mean: np.ndarray, pred_std: np.ndarray, kappa: float = 2.0
) -> np.ndarray:
    """Lower Confidence Bound acquisition for minimization: mu - kappa * sigma."""
    return pred_mean - kappa * pred_std


def plot_cross_section_and_acquisition(
    X1_grid: np.ndarray,
    X2_grid: np.ndarray,
    Z_true_grid: np.ndarray,
    preds: np.ndarray,
    X1_data: np.ndarray,
    X2_data: np.ndarray,
    Y_data: np.ndarray,
    kappa: float = 2.0,
):
    """Plot 2D predicted mean vs true and acquisition heatmap with sampled points shown.

    preds: shape (n_samps, n_grid_points)
    X1_grid/X2_grid: meshgrid arrays with shape (ny, nx)
    Z_true_grid: same shape
    """
    grid_shape = X1_grid.shape

    pred_mean = preds.mean(axis=0).reshape(grid_shape)
    pred_std = preds.std(axis=0).reshape(grid_shape)
    lcb = acquisition_lcb(pred_mean, pred_std, kappa=kappa)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: predicted mean contour with true contours overlaid and sampled points
    im1 = ax1.contourf(X1_grid, X2_grid, pred_mean, levels=50, cmap="viridis")
    ax1.contour(
        X1_grid,
        X2_grid,
        Z_true_grid,
        levels=8,
        colors="k",
        linewidths=0.8,
        linestyles="--",
    )
    sc = ax1.scatter(
        X1_data, X2_data, c=Y_data, cmap="autumn", edgecolor="k", s=40, label="Samples"
    )
    ax1.set_title("Predicted mean (with true contours and samples)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1)

    # Right: acquisition heatmap (LCB) — lower is better
    im2 = ax2.contourf(X1_grid, X2_grid, lcb, levels=50, cmap="plasma")
    # mark best point
    min_idx = np.argmin(lcb)
    iy, ix = np.unravel_index(min_idx, grid_shape)
    best_x = X1_grid[iy, ix]
    best_y = X2_grid[iy, ix]
    ax2.scatter([best_x], [best_y], color="red", s=60, label="Best (LCB)")
    ax2.set_title("Acquisition (LCB) — lower is better")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_before_after(
    X1_grid: np.ndarray,
    X2_grid: np.ndarray,
    Z_true_grid: np.ndarray,
    preds_before: np.ndarray,
    preds_after: np.ndarray,
    X1_data_before: np.ndarray,
    X2_data_before: np.ndarray,
    Y_before: np.ndarray,
    X1_data_after: np.ndarray,
    X2_data_after: np.ndarray,
    Y_after: np.ndarray,
    kappa: float = 2.0,
):
    """Plot predictions and acquisition before and after a single BO step on a grid.

    X1_grid/X2_grid are meshgrid arrays (ny, nx). preds_* are (n_samps, n_points).
    X1_data_before/X2_data_before are sample coords arrays.
    """
    grid_shape = X1_grid.shape

    mean_b = preds_before.mean(axis=0).reshape(grid_shape)
    std_b = preds_before.std(axis=0).reshape(grid_shape)
    lcb_b = acquisition_lcb(mean_b, std_b, kappa=kappa)

    mean_a = preds_after.mean(axis=0).reshape(grid_shape)
    std_a = preds_after.std(axis=0).reshape(grid_shape)
    lcb_a = acquisition_lcb(mean_a, std_a, kappa=kappa)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Prediction before
    im = axes[0, 0].contourf(X1_grid, X2_grid, mean_b, levels=50, cmap="viridis")
    axes[0, 0].scatter(
        X1_data_before, X2_data_before, c=Y_before, cmap="autumn", edgecolor="k", s=40
    )
    axes[0, 0].set_title("Prediction before")
    fig.colorbar(im, ax=axes[0, 0])

    # Prediction after
    im = axes[0, 1].contourf(X1_grid, X2_grid, mean_a, levels=50, cmap="viridis")
    axes[0, 1].scatter(
        X1_data_after, X2_data_after, c=Y_after, cmap="autumn", edgecolor="k", s=40
    )
    axes[0, 1].set_title("Prediction after")
    fig.colorbar(im, ax=axes[0, 1])

    # Acquisition before
    im = axes[1, 0].contourf(X1_grid, X2_grid, lcb_b, levels=50, cmap="plasma")
    axes[1, 0].set_title("Acquisition before")
    fig.colorbar(im, ax=axes[1, 0])

    # Acquisition after
    im = axes[1, 1].contourf(X1_grid, X2_grid, lcb_a, levels=50, cmap="plasma")
    # mark best after
    flat_idx = np.argmin(lcb_a)
    iy, ix = np.unravel_index(flat_idx, grid_shape)
    axes[1, 1].scatter(
        [X1_grid[iy, ix]], [X2_grid[iy, ix]], color="red", label="Best after"
    )
    axes[1, 1].set_title("Acquisition after")
    fig.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()


def main(
    n_initial: int = 5,
    draws: int = 600,
    tune: int = 400,
    x_range: float = 7.5,
    x_line_points: int = 400,
    y_const: float = None,
    kappa: float = 2.0,
    n_iter: int = 10,
):
    # 1. Generate a small initial dataset
    X1_data, X2_data, Y_data = gen_data(n=n_initial, x_range=5.0, y_const=y_const)

    # 2. Prepare a 2D grid (for visualization and acquisition over (x,y))
    nx = int(np.sqrt(x_line_points))
    ny = nx
    xs = np.linspace(-x_range, x_range, nx)
    ys = np.linspace(-x_range, x_range, ny)
    X1_grid, X2_grid = np.meshgrid(xs, ys)
    X_grid = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T
    Z_true_grid = rastrigin(X_grid, params).reshape(X1_grid.shape)

    prior_overrides = None

    # helper to build prior overrides from posterior
    def make_prior_overrides(trace):
        posterior = trace.posterior

        def mean_std(name):
            arr = posterior[name].values.reshape(-1, *posterior[name].shape[2:])
            vals = arr.mean(axis=0)
            sds = arr.std(axis=0)
            if vals.shape == ():
                return float(vals), float(max(sds, 1e-2))
            return float(vals), float(max(sds, 1e-2))

        return {
            "A0": mean_std("A0"),
            "A1": mean_std("A1"),
            "A2": mean_std("A2"),
            "omega1": mean_std("omega1"),
            "omega2": mean_std("omega2"),
            "chi1": mean_std("chi1"),
            "chi2": mean_std("chi2"),
        }

    # 3. Iterative BO loop
    all_X1 = list(X1_data)
    all_X2 = list(X2_data)
    all_Y = list(Y_data)

    preds_before = None
    trace = None
    for it in range(n_iter):
        print(f"\n=== Iteration {it+1}/{n_iter} ===")

        # a) fit with current data and prior_overrides
        model = build_model(
            np.array(all_X1),
            np.array(all_X2),
            np.array(all_Y),
            prior_overrides=prior_overrides,
        )
        trace = sample_posterior(model, draws=draws, tune=tune, target_accept=0.95)

        # b) save and print summary
        summary = az.summary(trace)
        fname = f"trace_summary_iter_{it+1}.csv"
        try:
            summary.to_csv(fname)
            print(f"Saved summary to {fname}")
        except Exception as e:
            print("Failed to save summary:", e)
        print(summary)

        # c) predictive on grid
        param_samps = posterior_param_samples(trace)
        preds = predictive_samples_on_grid(X_grid, param_samps, max_samps=200)

        # store preds_before for first iter for optional plotting
        if it == 0:
            preds_before = preds

        # d) acquisition and pick best
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        lcb = acquisition_lcb(mean, std, kappa=kappa)
        best_idx = int(np.argmin(lcb))
        best_xy = X_grid[best_idx]

        # e) evaluate and append
        y_new = noisy_rastrigin(best_xy.reshape(1, 2), params)[0]
        all_X1.append(float(best_xy[0]))
        all_X2.append(float(best_xy[1]))
        all_Y.append(float(y_new))

        # f) set prior_overrides from posterior for next iter
        prior_overrides = make_prior_overrides(trace)

    # end loop

    # final predictive samples on grid
    final_param_samps = posterior_param_samples(trace)
    preds_final = predictive_samples_on_grid(X_grid, final_param_samps, max_samps=500)

    # 4. Show final 2D plots
    plot_cross_section_and_acquisition(
        X1_grid,
        X2_grid,
        Z_true_grid,
        preds_final,
        np.array(all_X1),
        np.array(all_X2),
        np.array(all_Y),
        kappa=kappa,
    )

    # Optional before/after visualization using preds_before and preds_final
    if preds_before is not None:
        plot_before_after(
            X1_grid,
            X2_grid,
            Z_true_grid,
            preds_before,
            preds_final,
            np.array(X1_data),
            np.array(X2_data),
            np.array(Y_data),
            np.array(all_X1),
            np.array(all_X2),
            np.array(all_Y),
            kappa=kappa,
        )


if __name__ == "__main__":
    main()
