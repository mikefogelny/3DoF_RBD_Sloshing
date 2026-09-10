"""
Example/test script: 10 attitude-maneuver trajectories on one shared
satellite configuration.

Covers, in order:
    1-3  Single-axis maneuvers (roll-only, pitch-only, yaw-only)
    4-6  Two-axis coupled maneuvers (roll+pitch, roll+yaw, pitch+yaw)
    7    All three axes simultaneously (max coupling)
    8    Same magnitude as #7 but opposite direction (checks symmetry)
    9    Large single-axis maneuver near 180 deg (stresses the
         sign(dq4) shortest-path logic in pd_controller)
    10   Small-angle target reached from a nonzero initial body-rate
         (tests near-linear response under an existing disturbance)

Uses rigid_body_pd_v3's run_batch() to hold inertia/gains/slosh model fixed
across all 10 runs, varying only q0/q_des/w0 per scenario.
"""

import numpy as np
import matplotlib.pyplot as plt

from rigid_body_pd_v3 import (
    mc_example_72, euler321_to_quat,
    run_batch, plot_batch, export_batch_to_mat,
)


def build_scenarios():
    """Return the list of 10 scenario dicts (q0/q_des, and w0 for #10)."""
    zero = euler321_to_quat(0, 0, 0)

    scenarios = [
        # 1-3: single-axis
        {'q0': zero, 'q_des': euler321_to_quat(90, 0, 0)},
        {'q0': zero, 'q_des': euler321_to_quat(0, 90, 0)},
        {'q0': zero, 'q_des': euler321_to_quat(0, 0, 90)},
        # 4-6: two-axis coupled
        {'q0': zero, 'q_des': euler321_to_quat(90, 90, 0)},
        {'q0': zero, 'q_des': euler321_to_quat(90, 0, 90)},
        {'q0': zero, 'q_des': euler321_to_quat(0, 90, 90)},
        # 7: all three axes simultaneously
        {'q0': zero, 'q_des': euler321_to_quat(90, 90, 90)},
        # 8: same magnitude, opposite direction
        {'q0': zero, 'q_des': euler321_to_quat(-90, -90, -90)},
        # 9: large single-axis maneuver near 180 deg
        {'q0': zero, 'q_des': euler321_to_quat(180, 0, 0)},
        # 10: small-angle target, nonzero initial tumble
        {'q0': zero, 'q_des': euler321_to_quat(10, 10, 10),
         'w0': np.array([0.05, -0.03, 0.02])},
    ]
    return scenarios


def main():
    base_cfg = mc_example_72()
    base_cfg.t_end = 120.0     # long enough to see each maneuver settle

    scenarios = build_scenarios()

    print(f"=== Running {len(scenarios)} trajectories ===")
    batch_results = run_batch(base_cfg, scenarios)

    plot_batch(batch_results, scenarios)
    plt.show()

    export_batch_to_mat(batch_results, scenarios, base_cfg,
                        filepath="ten_trajectories.mat")


if __name__ == "__main__":
    main()
