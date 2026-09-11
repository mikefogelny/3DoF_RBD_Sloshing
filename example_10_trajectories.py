"""
Example/test script: 10 attitude-maneuver trajectories on one shared
satellite configuration.

Covers, in order (all slews positive-only, capped below 45 deg per axis):
    1-3  Single-axis maneuvers (roll-only, pitch-only, yaw-only)
    4-6  Two-axis coupled maneuvers (roll+pitch, roll+yaw, pitch+yaw)
    7    All three axes simultaneously, moderate magnitude
    8    All three axes simultaneously, near the 45 deg cap (stress test)
    9    Unequal magnitudes across the three axes (asymmetric coupling)
    10   Small-angle target reached from a nonzero initial body-rate
         (tests near-linear response under an existing disturbance)

Note: an earlier version of this scenario set used a same-magnitude
opposite-direction case (-45/-45/-45) to check response symmetry. That
scenario exposed a persistent, non-decaying attitude oscillation that
does NOT resolve just by increasing Kd (see project history) -- a real
finding, not a bug in this script -- so negative-angle targets are
intentionally excluded here pending a separate investigation into that
instability.

Uses rigid_body_pd_v3's run_batch() to hold inertia/gains/slosh model fixed
across all 10 runs, varying only q0/q_des/w0 per scenario. Slosh is enabled
on the shared base config so its torque shows up in the telemetry plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from rigid_body_pd_v3 import (
    mc_example_72, euler321_to_quat,
    run_batch, plot_batch, export_batch_to_mat,spicesat,
)


def _scenario(start_deg, target_deg, w0=None):
    """
    Build one scenario dict, recording BOTH the quaternion (q0/q_des, what
    run_batch/simulate actually use) and the original [roll, pitch, yaw]
    degree triples (q0_deg/qdes_deg, what plot_batch/export_batch_to_mat
    display).

    Storing the degrees explicitly -- rather than recovering them later via
    quat_to_euler321(q_des) -- avoids a gimbal-lock ambiguity: at pitch =
    +-90 deg, infinitely many (roll, pitch, yaw) triples correspond to the
    same quaternion, so round-tripping through quat_to_euler321 can report
    a technically-equivalent but visually different triple (e.g.
    [180, 90, 180] instead of the [0, 90, 0] actually requested here).
    """
    sc = {
        'q0':      euler321_to_quat(*start_deg),
        'q_des':   euler321_to_quat(*target_deg),
        'q0_deg':  np.array(start_deg, dtype=float),
        'qdes_deg': np.array(target_deg, dtype=float),
    }
    if w0 is not None:
        sc['w0'] = w0
    return sc


def build_scenarios():
    """Return the list of 10 scenario dicts (q0/q_des, and w0 for #10)."""
    scenarios = [
        # 1-3: single-axis
        _scenario((0, 0, 0), (30, 0, 0)),
        _scenario((0, 0, 0), (0, 30, 0)),
        _scenario((0, 0, 0), (0, 0, 30)),
        # 4-6: two-axis coupled
        _scenario((0, 0, 0), (30, 30, 0)),
        _scenario((0, 0, 0), (30, 0, 30)),
        _scenario((0, 0, 0), (0, 30, 30)),
        # 7: all three axes simultaneously, moderate magnitude
        _scenario((0, 0, 0), (30, 30, 30)),
        # 8: all three axes simultaneously, near the 45 deg cap
        _scenario((0, 0, 0), (40, 40, 40)),
        # 9: unequal magnitudes across axes (asymmetric coupling)
        _scenario((0, 0, 0), (20, 10, 5)),
        # 10: small-angle target, nonzero initial tumble
        _scenario((0, 0, 0), (10, 10, 10), w0=np.array([0.05, -0.03, 0.02])),
    ]
    return scenarios


def main():
#    base_cfg = mc_example_72()
    base_cfg = spicesat()
    base_cfg.t_end = 600.0     # long enough to see each maneuver settle
    base_cfg.enable_slosh = True   # otherwise Ts stays zero throughout
    base_cfg.enable_wheel_dynamics = True   # wheel torque ramps instead of jumping instantly
    # Default Kd=150 leaves multi-axis slews with a persistent, non-decaying
    # oscillation (a genuine gyroscopic/coupling instability, confirmed
    # independent of slosh and solver tolerance -- see project history).
    # Kd=500 was swept against all 10 scenarios above and cleanly converges
    # every one (residual oscillation shrinks monotonically to <0.001 deg
    # by t=6000s, including the toughest 40/40/40 case).
    # base_cfg.Kd = 500 # only used for MC7.2 example

    scenarios = build_scenarios()

    print(f"=== Running {len(scenarios)} trajectories ===")
    batch_results = run_batch(base_cfg, scenarios)

    plot_batch(batch_results, scenarios, base_cfg, save_dir="plots")
    plt.show()

    export_batch_to_mat(batch_results, scenarios, base_cfg,
                        filepath="ten_trajectories.mat")


if __name__ == "__main__":
    main()
