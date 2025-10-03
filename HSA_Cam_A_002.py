import typing as t

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor as pt
from scipy import stats

from data.rastrigin import rastrigin, noisy_rastrigin, params


def gen_data(n: int = 100, x_range: float = 5.0, y_const: t.Optional[float] = None):
    """Generate n 2D sample locations and noisy observations of the rastrigin function.

    Returns X, Y arrays.
    """
    X = np.random.uniform(-x_range, x_range, size=(n, 2))
    if y_const is not None:
        X[:, 1] = y_const
    return X.copy(), noisy_rastrigin(X, params)


def pt_rastrigin(X, params):
    A0 = params[0]
    A1 = params[1]
    A2 = params[2]
    omega1 = params[3]
    omega2 = params[4]
    chi1 = params[5]
    chi2 = params[6]
    val = (
        A0
        + ((X[:, 0] - chi1) ** 2 - A1 * pt.tensor.cos(omega1 * (X[:, 0] - chi1)))
        + ((X[:, 1] - chi2) ** 2 - A2 * pt.tensor.cos(omega2 * (X[:, 1] - chi2)))
    )
    return val


def _from_posterior(name: str, idata: az.InferenceData) -> pm.Interpolated:
    samples = az.extract(idata, group="posterior", var_names=[name])
    smin, smax = samples.min().item(), samples.max().item()
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(name, x, y)


def build_model(X, Y, prior: t.Optional[az.InferenceData] = None):
    with pm.Model() as model:
        if prior is None:
            # Priors for unknown model parameters
            A0 = pm.Normal("A0", mu=35, sigma=20)
            A1 = pm.Normal("A1", mu=10, sigma=3)
            A2 = pm.Normal("A2", mu=10, sigma=3)
            omega1 = pm.LogNormal("omega1", mu=0.75, sigma=0.15)
            omega2 = pm.LogNormal("omega2", mu=0.75, sigma=0.15)
            chi1 = pm.Normal("chi1", mu=0, sigma=np.pi / 2)
            chi2 = pm.Normal("chi2", mu=0, sigma=np.pi / 2)
            sigma = pm.HalfNormal("sigma", sigma=10)

        else:
            A0 = _from_posterior("A0", prior)
            A1 = _from_posterior("A1", prior)
            A2 = _from_posterior("A2", prior)
            omega1 = _from_posterior("omega1", prior)
            omega2 = _from_posterior("omega2", prior)
            chi1 = _from_posterior("chi1", prior)
            chi2 = _from_posterior("chi2", prior)
            sigma = _from_posterior("sigma", prior)

        mu = pt_rastrigin(X, [A0, A1, A2, omega1, omega2, chi1, chi2])

        pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)
    return model


def main():
    X_data, Y_data = gen_data(n=10, x_range=10)
    model = build_model(X_data, Y_data)
    with model:
        idata = pm.sample(
            2000,
            tune=400,
            return_inferencedata=True,
            target_accept=0.95,
            max_treedepth=12,
        )
    print(
        az.summary(
            idata,
            var_names=["A0", "A1", "A2", "omega1", "omega2", "chi1", "chi2", "sigma"],
        )
    )
    az.plot_trace(
        idata,
        var_names=["A0", "A1", "A2", "omega1", "omega2", "chi1", "chi2", "sigma"],
    )
    plt.show()

    # add another 10 data points and sample again with prior from previous run
    X_data, Y_data = gen_data(n=10, x_range=10)
    model = build_model(X_data, Y_data, prior=idata)
    with model:
        new_idata = pm.sample(
            2000,
            tune=400,
            return_inferencedata=True,
            target_accept=0.95,
            max_treedepth=12,
        )

    print(
        az.summary(
            new_idata,
            var_names=["A0", "A1", "A2", "omega1", "omega2", "chi1", "chi2", "sigma"],
        )
    )
    az.plot_trace(
        idata,
        var_names=["A0", "A1", "A2", "omega1", "omega2", "chi1", "chi2", "sigma"],
        divergences=False,
        combined=True,
    )
    ax = plt.gca()
    az.plot_trace(
        new_idata,
        var_names=["A0", "A1", "A2", "omega1", "omega2", "chi1", "chi2", "sigma"],
        divergences=False,
        ax=ax,
    )
    plt.show()


if __name__ == "__main__":
    main()
