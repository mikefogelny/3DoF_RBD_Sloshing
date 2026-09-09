"""
Smoke test for rigid_body_pd_v3.py's multi-trajectory batch support.

Runs 9 scenarios on one shared base SimConfig (fixed inertia, wheel
geometry, gains, slosh model), all starting at Euler angles [0, 0, 0] deg
and commanded to [90, 90, 90] deg, differing only in run-to-run numerical
noise-free reproducibility (i.e. this exercises run_batch()'s per-scenario
config isolation, not distinct physical scenarios).
"""

import sys
import numpy as np

from rigid_body_pd_v3 import (
    mc_example_72, run_batch, export_batch_to_mat,
    euler321_to_quat,
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def main():
    all_ok = True

    base_cfg = mc_example_72()
    base_cfg.t_end = 60.0     # keep the smoke test fast

    q0    = euler321_to_quat(0, 0, 0)
    q_des = euler321_to_quat(90, 90, 90)
    scenarios = [{'q0': q0, 'q_des': q_des} for _ in range(9)]

    print(f"=== Running batch of {len(scenarios)} scenarios ===")
    batch_results = run_batch(base_cfg, scenarios)

    all_ok &= check("9 results returned", len(batch_results) == 9)

    # Base config must be untouched by run_batch's per-scenario copies.
    all_ok &= check(
        "base_cfg.q0 unmodified",
        np.allclose(base_cfg.q0, mc_example_72().q0))

    for i, results in enumerate(batch_results):
        t, q_hist, dq_hist, w_hist = results[0], results[1], results[2], results[3]
        all_ok &= check(f"traj {i}: no NaNs", not np.any(np.isnan(q_hist)))
        norms = np.linalg.norm(q_hist, axis=1)
        all_ok &= check(f"traj {i}: |q| ~= 1",
                        np.allclose(norms, 1.0, atol=1e-6))

    # All 9 trajectories used identical scenarios -> results must match.
    ref_q = batch_results[0][1]
    for i, results in enumerate(batch_results[1:], start=1):
        all_ok &= check(f"traj {i} matches traj 0 (identical scenarios)",
                        np.allclose(results[1], ref_q))

    export_batch_to_mat(batch_results, scenarios, base_cfg,
                        filepath="test_batch_export.mat")

    from scipy.io import loadmat
    mat = loadmat("test_batch_export.mat")
    trajectories = mat["trajectories"]
    all_ok &= check("trajectories shape is (1, 9)",
                    trajectories.shape == (1, 9), f"{trajectories.shape}")
    first = trajectories[0, 0]
    all_ok &= check("first trajectory has 19 data columns",
                    first["data"][0, 0].shape[1] == 19,
                    f"{first['data'][0, 0].shape}")

    import os
    os.remove("test_batch_export.mat")

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
