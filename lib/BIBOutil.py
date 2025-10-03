import arviz as az
import pymc as pm
import numpy as np
import scipy.stats as stats
from typing import Optional


def _from_posterior(
    name: str,
    idata: az.InferenceData,
    range: Optional[tuple[float | None, float | None]] = None,
    **kwargs,
) -> pm.Interpolated:
    samples = az.extract(idata, group="posterior", var_names=[name])
    smin, smax = samples.min().item(), samples.max().item()
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    match range:
        case (None, None):
            x = np.concatenate([[smin - 3 * width], x, [smax + 3 * width]])
            y = np.concatenate([[0], y, [0]])
        case (None, float(max)):
            x = np.concatenate([[smin - 3 * width], x, [max]])
            y = np.concatenate([[0], y, [0]])
        case (float(min), None):
            x = np.concatenate([[min], x, [smax + 3 * width]])
            y = np.concatenate([[0], y, [0]])
        case (float(min), float(max)):
            x = np.concatenate([[min], x, [max]])
            y = np.concatenate([[0], y, [0]])
        case None:
            pass

    # x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    # y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(name, x, y, **kwargs)
