import json
import argparse
from pathlib import Path
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

import pymc as pm
import pytensor

import arviz as az
import xarray
import namer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workers",
    type=int,
    default=8,
    help="Number of workers for parallel processing",
)
parser.add_argument(
    "--KLsamples",
    type=int,
    default=8,
    help="Number of hypothetical samples for KL divergence estimation",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=20,  # TODO
    help="Number of points to additionally sample in the experiment",
)

from HSA_Cam_B_013 import (
    parser,
    TRUE_PARAMS,
    bmodel,
    post_model_at_proposal,
    observe_at_proposal,
    mp_KL_divergence,
    kl_divergence_between_models,
)


def FFD():
    # Define the parameter ranges for the full factorial design
    T_range = np.array([500, 600])
    p_range = np.array([2, 20])
    c2h_range = np.array([0.5, 2])
    f_range = np.array([1, 1000])
    c2i_range = np.array([0.1, 1])

    # Create the full factorial design grid
    Tgrid, pgrid, c2hgrid, fgrid, c2igrid = np.meshgrid(
        T_range, p_range, c2h_range, f_range, c2i_range
    )

    # Combine the grids into a list of proposals
    proposals = jnp.vstack(
        [Tgrid.ravel(), pgrid.ravel(), c2hgrid.ravel(), fgrid.ravel(), c2igrid.ravel()]
    ).T

    # reorder for better space-filling
    proposals_new = proposals[
        [
            2,
            10,
            18,
            26,
            6,
            14,
            22,
            30,
            3,
            11,
            19,
            27,
            1,
            9,
            17,
            25,
            0,
            8,
            16,
            24,
            4,
            12,
            20,
            28,
            5,
            13,
            21,
            29,
            7,
            15,
            23,
            31,
        ],
        :,
    ]
    return proposals_new


def main(args):
    num_workers = args.workers
    n_hyp_samples = args.KLsamples
    # n_iterations = args.iterations

    full_proposals = FFD()
    proposals = np.empty((0, 5))
    r_obs = np.empty((0, 1))

    ## get the prior for the first information gain
    T, p, c2h, f, c2i, full_r_obs = observe_at_proposal(full_proposals)
    model = bmodel(
        (
            np.asarray(T),
            np.asarray(p),
            np.asarray(c2h),
            np.asarray(f),
            np.asarray(c2i),
        ),
        np.asarray(full_r_obs),
    )
    with model:
        idata = pm.sample_prior_predictive(samples=1000)
    idata["posterior"] = idata.prior

    trials_sum = []

    for trial_idx in range(full_proposals.shape[0] + 1):
        print(f"Trial {trial_idx}/{full_proposals.shape[0]}")

        trial_sum = {
            "iteration": trial_idx,
            "observed": {
                "x": proposals.tolist() if trial_idx > 0 else None,
                "y": r_obs.tolist() if trial_idx > 0 else None,
                "KLD": None,
            },
        }

        if trial_idx > 0:
            # actually build and infere the model, otherwise use the prior dummy
            T, p, c2h, f, c2i, r_obs = observe_at_proposal(proposals)
            model = bmodel(
                (
                    np.asarray(T),
                    np.asarray(p),
                    np.asarray(c2h),
                    np.asarray(f),
                    np.asarray(c2i),
                ),
                np.asarray(r_obs),
            )
            old_idata = idata
            with model:
                idata = pm.sample_prior_predictive(draws=1000)
                idata.extend(
                    pm.sample(
                        tune=1000,
                        draws=4000,
                        chains=12,
                        cores=12,
                        target_accept=0.9,
                        return_inferencedata=True,
                    )
                )
            obs_KL_vals = kl_divergence_between_models(
                idata_new=idata,
                idata_old=old_idata,
                n_samples=1000,
            )
            trial_sum["observed"]["KLD"] = obs_KL_vals.sum(axis=1).mean(axis=0).item()
        # at this point idata is either the dummy if i==0
        # or the actual inference if i>0
        idata.to_netcdf(save_path / f"trace_{trial_idx:03d}.nc")
        trial_sum.update(
            {
                "trace_path": str(save_path / f"trace_{trial_idx:03d}.nc"),
            }
        )

        # check if we still have proposals to process
        if trial_idx < full_proposals.shape[0]:
            # one proposals left to process
            # predict KLD for appending and then append
            # We dont need to maximize KLD, as the next experiment is fixed
            proposal = full_proposals[trial_idx : trial_idx + 1, :]
            KLD_vals = mp_KL_divergence(
                proposals,
                r_obs,
                proposal.squeeze(),
                idata,
                n_hyp_samples,
                num_workers,
            )
            trial_sum.update(
                {
                    "expected": {
                        "x": proposal.tolist(),
                        "KLD": KLD_vals.sum(axis=1).mean(axis=0).item(),
                    }
                }
            )
            proposals = np.append(proposals, np.atleast_2d(proposal), axis=0)
        else:
            trial_sum.update(
                {
                    "expected": {
                        "x": None,
                        "KLD": None,
                    }
                }
            )
        trials_sum.append(trial_sum)
        exp_summary.update({"trials": trials_sum})
        with open(save_path / f"experiment_summary_{EXP_ID}.json", "w") as f:
            f.write(json.dumps(exp_summary))


if __name__ == "__main__":
    args = parser.parse_args()
    global EXP_ID, exp_summary, save_path
    EXP_ID = namer.generate()
    print(f"===========\nExperiment ID: {EXP_ID}\n===========")
    exp_summary = {
        "experiment_id": EXP_ID,
        "root file": str(Path.cwd()),
        "args": vars(args),
    }
    save_path = Path.cwd() / "data" / f"{EXP_ID}"
    save_path.mkdir()
    main(args)
    with open(save_path / f"experiment_summary_{EXP_ID}.json", "w") as f:
        f.write(json.dumps(exp_summary))
