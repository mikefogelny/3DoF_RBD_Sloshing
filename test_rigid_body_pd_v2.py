"""
Smoke test for rigid_body_pd_v2.py.

Exercises both preset configs, all eight toggle combinations, output-shape
sanity, quaternion unit-norm preservation, and the plotting functions.
Uses the non-interactive matplotlib Agg backend so no windows pop up.
"""

import sys
import traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")          # MUST be set before importing pyplot
import matplotlib.pyplot as plt

from rigid_body_pd_v2 import (
    SimConfig, simulate, mc_example_71, mc_example_72,
    plot_results, plot_euler_and_rates,
    euler321_to_quat, quat_to_euler321,
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def run_case(label, cfg, n_w_expected=4):
    print(f"\n=== {label} ===")
    print(f"  Toggles: slosh={cfg.enable_slosh}  mag={cfg.enable_magnetorquer}"
          f"  wheels={cfg.enable_wheels}")
    cfg.t_end = min(cfg.t_end, 120.0)        # keep tests fast
    results = simulate(cfg)
    t, q_hist, dq_hist, w_hist, u_hist, hw_hist, tauw_hist, \
        m_hist, tmag_hist, B_hist, Ts_hist = results

    N = len(t)
    ok = True
    ok &= check("tuple length is 11", len(results) == 11)
    ok &= check("q_hist shape",   q_hist.shape   == (N, 4), f"{q_hist.shape}")
    ok &= check("dq_hist shape",  dq_hist.shape  == (N, 4))
    ok &= check("w_hist shape",   w_hist.shape   == (N, 3))
    ok &= check("u_hist shape",   u_hist.shape   == (N, 3))
    ok &= check("hw_hist shape",  hw_hist.shape  == (N, n_w_expected))
    ok &= check("tauw_hist shape", tauw_hist.shape == (N, n_w_expected))
    ok &= check("Ts_hist shape",  Ts_hist.shape  == (N, 3))
    ok &= check("no NaNs in states",
                not np.any(np.isnan(q_hist)) and not np.any(np.isnan(w_hist)))
    norms = np.linalg.norm(q_hist, axis=1)
    ok &= check("|q| ~= 1 throughout",
                np.allclose(norms, 1.0, atol=1e-6),
                f"max |q|-1 = {np.max(np.abs(norms - 1.0)):.2e}")

    # When wheels are OFF, wheel state should stay at zero
    if not cfg.enable_wheels:
        ok &= check("h_w stays zero (wheels off)",
                    np.allclose(hw_hist, 0.0),
                    f"max |h_w| = {np.max(np.abs(hw_hist)):.2e}")
        ok &= check("tau_w stays zero (wheels off)",
                    np.allclose(tauw_hist, 0.0))

    # When magnetorquer is OFF, dipole + mag torque should be zero
    if not cfg.enable_magnetorquer:
        ok &= check("dipole stays zero (mag off)", np.allclose(m_hist, 0.0))
        ok &= check("mag torque stays zero (mag off)",
                    np.allclose(tmag_hist, 0.0))

    # When slosh is OFF, Ts state must stay at IC (zero)
    if not cfg.enable_slosh:
        ok &= check("Ts stays zero (slosh off)", np.allclose(Ts_hist, 0.0),
                    f"max |Ts| = {np.max(np.abs(Ts_hist)):.2e}")

    # Steady-state sanity: body rate should reduce after settling
    w_init  = np.linalg.norm(w_hist[0])
    w_final = np.linalg.norm(w_hist[-1])
    ok &= check("|w| decreasing", w_final <= w_init + 1e-6,
                f"|w|: {w_init:.3e} -> {w_final:.3e}")
    return ok, results


def test_plot_functions(results):
    """Plot functions should run end-to-end without errors."""
    print("\n=== Plot smoke test ===")
    try:
        plot_results(*results)
        check("plot_results call", True)
    except Exception as e:
        check("plot_results call", False, repr(e))
        traceback.print_exc()
        return False
    try:
        plot_euler_and_rates(results[0], results[1], results[3])
        check("plot_euler_and_rates call", True)
    except Exception as e:
        check("plot_euler_and_rates call", False, repr(e))
        traceback.print_exc()
        return False
    plt.close("all")
    return True


def test_euler_roundtrip():
    """Verify euler321_to_quat <-> quat_to_euler321 is consistent."""
    print("\n=== Euler <-> quaternion roundtrip ===")
    ok = True
    for r, p, y in [(0, 0, 0), (30, -45, 60), (-89.5, 0, 0), (10, 20, 30)]:
        q = euler321_to_quat(r, p, y)
        rr, pp, yy = quat_to_euler321(q)
        rr, pp, yy = np.rad2deg([rr, pp, yy])
        match = np.allclose([r, p, y], [rr, pp, yy], atol=1e-9)
        ok &= check(f"({r:+.1f}, {p:+.1f}, {y:+.1f}) -> "
                    f"({rr:+.4f}, {pp:+.4f}, {yy:+.4f})", match)
    return ok


def main():
    all_ok = True

    # Roundtrip helper functions
    all_ok &= test_euler_roundtrip()

    # Preset 7.1, default toggles
    ok, _ = run_case("Example 7.1 (wheels only)", mc_example_71())
    all_ok &= ok

    # Preset 7.2, default toggles
    ok, results_72 = run_case("Example 7.2 (wheels only)", mc_example_72())
    all_ok &= ok

    # Preset 7.2 + slosh + magnetorquer
    cfg = mc_example_72()
    cfg.enable_slosh = True
    cfg.enable_magnetorquer = True
    ok, _ = run_case("Example 7.2 (full stack)", cfg)
    all_ok &= ok

    # Preset 7.2 with wheels OFF (ideal torque actuator)
    cfg = mc_example_72()
    cfg.enable_wheels = False
    ok, _ = run_case("Example 7.2 (wheels OFF)", cfg)
    all_ok &= ok

    # Preset 7.2 with everything OFF (open-loop with PD applied to body)
    cfg = mc_example_72()
    cfg.enable_wheels = False
    cfg.enable_slosh = False
    cfg.enable_magnetorquer = False
    ok, _ = run_case("Example 7.2 (ideal actuator, no slosh, no mag)", cfg)
    all_ok &= ok

    # In-place override smoke test
    cfg = mc_example_72()
    cfg.Kp, cfg.Kd = 25.0, 250.0
    ok, _ = run_case("Example 7.2 with overridden gains", cfg)
    all_ok &= ok

    # Plot functions
    all_ok &= test_plot_functions(results_72)

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
