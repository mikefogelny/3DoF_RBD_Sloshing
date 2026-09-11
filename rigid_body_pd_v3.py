"""
Rigid-body attitude dynamics with quaternion PD control, reaction wheels,
magnetic momentum dumping, and a propellant-slosh disturbance torque.

Version 2 — refactored so that all spacecraft parameters, gains, initial
conditions, time settings, and toggles live in a single SimConfig dataclass
at the top of the file.  Named preset factory functions are provided for the
worked examples in Markley & Crassidis (Examples 7.1 and 7.2).

Version 3 — adds multi-trajectory batch support: run_batch() reuses a single
base SimConfig (fixed inertia, wheel geometry, gains, slosh model) across a
list of scenario overrides (q0 / q_des / w0 per trajectory), and
export_batch_to_mat() exports all trajectories to one MATLAB .mat file.

Integrator: scipy.integrate.solve_ivp (adaptive RK45 by default).
Convention: SCALAR-LAST quaternion  q = [q1, q2, q3, q4],  q4 = scalar.

Reference:
    Markley, F.L. and Crassidis, J.L., "Fundamentals of Spacecraft Attitude
    Determination and Control" (Springer, 2014).
        - Eq. 2.82b  : quaternion product, scalar-last
        - Eq. 2.88   : quaternion kinematics, scalar-last
        - Eq. 3.13   : inertia tensor sign convention
        - Sec. 7.2   : reaction-wheel dynamics
        - Sec. 7.4   : quaternion-feedback PD control
        - Sec. 7.5   : magnetic momentum unloading (cross-product law)
        - App. B.96  : 3-2-1 Euler-angle <-> quaternion conversion
"""

import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from scipy.signal import tf2ss


# =====================================================================
# Configuration  ── all knobs in one place; edit here to change the sim
# =====================================================================

@dataclass
class SimConfig:
    """
    All satellite parameters, gains, ICs, time, and toggles.

    This is the single source of truth for a simulation "scenario". Every
    knob that could plausibly change between runs lives here rather than as
    a hard-coded constant elsewhere in the file, so that:
      - `simulate(cfg)` is a pure function of its input config (no hidden
        global state), and
      - `run_batch()` can vary just a few fields (e.g. q0, q_des) while
        holding everything else (inertia, gains, slosh model) fixed by
        deep-copying one shared base config per scenario.
    """

    # ---- Inertia tensor components (M&C Eq. 3.13), kg·m^2 ----
    # These are the physical mass distribution of the spacecraft. Ixx/Iyy/Izz
    # are the principal-axis-ish diagonal moments; Ixy/Ixz/Iyz are the
    # (positive-valued) products of inertia — the sign flip needed to build
    # the actual inertia tensor J happens later in simulate(), not here.
    Ixx: float = 6400.0
    Iyy: float = 4730.0
    Izz: float = 8160.0
    Ixy: float = 76.4        # products of inertia (signs handled in J builder)
    Ixz: float = 25.6
    Iyz: float = 40.0

    # ---- Reaction-wheel geometry (4-wheel NASA pyramid) ----
    # Four wheels are mounted with their spin axes tilted this many degrees
    # off the body +Z axis, spaced 90 deg apart in azimuth. This is a
    # standard redundant "pyramid" layout: any 3 of the 4 wheels can fully
    # actuate the spacecraft if one fails. The actual 3x4 mapping matrix W
    # is built from this angle inside simulate().
    wheel_tilt_deg: float = 54.7356      # acos(1/sqrt(3))

    # ---- Control gains and saturation ----
    Kp:     float = 10.0     # proportional gain (attitude error -> torque)
    Kd:     float = 150.0    # derivative gain (body rate -> damping torque)
    u_max:  float = 1000.0   # N·m,    per-axis PD torque saturation
    K_dump: float = 1.0      # 1/s,    magnetic momentum-dump gain
    m_max:  float = 500.0    # A·m^2,  per-axis magnetorquer dipole saturation

    # ---- Reaction-wheel actuator dynamics (2nd-order low-pass, per wheel) ----
    # Real wheels don't respond instantly -- they have finite spin-up/down
    # dynamics. Modeled here as a transfer function
    #     tau_actual(s) / tau_cmd(s) = wheel_tf_num(s) / wheel_tf_den(s)
    # applied independently per wheel. Coefficients are in descending powers
    # of s (scipy.signal convention), so changing the actuator response is
    # just editing these two coefficient lists -- any filter order, no
    # manual state-space/canonical-form derivation needed (see
    # wheel_actuator_dynamics(), which builds the state-space realization
    # automatically via scipy.signal.tf2ss). Default: instantaneous
    # response (filter bypassed) unless enabled.
    enable_wheel_dynamics: bool = False
    wheel_tf_num: list = field(default_factory=lambda: [1.2, 0.76])       # 1.2*s + 0.76
    wheel_tf_den: list = field(default_factory=lambda: [1.0, 2.4, 0.76])  # s^2 + 2.4*s + 0.76

    # ---- Slosh model selector: 'bourdelle' (default) or 'testing' ----
    # Two interchangeable second-order ODEs describe how sloshing propellant
    # reacts back on the spacecraft body as a disturbance torque Ts. Only
    # ONE is active per run — dynamics_rhs() branches on this string.
    # 'bourdelle' = the original slosh_dynamics_bourdelle() model.
    # 'testing'   = the newer rate-dependent slosh_dynamics_testing() model.
#    slosh_model: str = 'bourdelle'
    slosh_model: str = 'testing'


    # ---- Slosh model (bourdelle):  f(t) = nom + amp*sin(omega*t) ----
    # Each of the four scalar coefficients (A, B, C, K) in the slosh ODE can
    # itself vary sinusoidally in time around a nominal (DC) value — see
    # slosh_coefficient(). With amp=0 (the default here) each coefficient is
    # just a constant equal to 'nom'.
    slosh_params: dict = field(default_factory=lambda: {
        'A': {'nom': 0.005, 'amp': 0.0, 'omega': 0.05},
        'B': {'nom': 0.020, 'amp': 0.0, 'omega': 0.05},
        'C': {'nom': 0.023, 'amp': 0.0, 'omega': 0.05},
        'K': {'nom': 0.008, 'amp': 0.0, 'omega': 0.05},
    })

    # ---- Slosh model (testing): rate-dependent stiffness/damping ----
    # Parameters for slosh_dynamics_testing(): natural frequency and damping
    # ratio of the sloshing mode, plus a saturation rate omega_max used to
    # scale the effective stiffness/damping as the body spin rate grows.
    slosh_testing_params: dict = field(default_factory=lambda: {
        'omega_n':   0.08944,    # rad/s, sloshing-mode natural frequency
        'zeta':      0.1286,     # sloshing-mode damping ratio
        'omega_max': 0.418879,   # rad/s, max expected body rotation rate (SPICEsat: 24 deg/s)
    })

    # ---- Initial conditions ----
    # q0 / w0 describe how the spacecraft starts (attitude quaternion and
    # body rate); q_des is the commanded target attitude the PD controller
    # tries to drive the spacecraft toward. Ts0/Tsd0 are the initial slosh
    # torque and its rate — usually zero (propellant starts undisturbed).
    q0:    np.ndarray = field(default_factory=lambda:
        (math.sqrt(2) / 2.0) * np.array([1.0, 0.0, 0.0, 1.0]))   # 90° about +X
    w0:    np.ndarray = field(default_factory=lambda:
        np.array([0.01, 0.01, 0.01]))                            # rad/s
    q_des: np.ndarray = field(default_factory=lambda:
        np.array([0.0, 0.0, 0.0, 1.0]))                          # target = identity
    Ts0:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    Tsd0:  np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ---- Time grid ----
    t_end:   float = 20 * 60.0    # seconds; total simulated duration
    dt_eval: float = 0.05         # output sample step (NOT the solver's internal step)

    # ---- Toggles ----
    # These let you isolate individual subsystems for testing/debugging.
    # e.g. with enable_wheels=False, the PD command drives the body directly
    # as an ideal torque actuator (see dynamics_rhs, actuator branch).
    enable_slosh:        bool = False
    enable_magnetorquer: bool = False
    enable_wheels:       bool = True

    # ---- Solver settings ----
    # Passed straight through to scipy.integrate.solve_ivp. max_step is
    # capped fairly low (0.5 s) because the slosh model and magnetorquer law
    # both have their own fast timescales that a fully adaptive step could
    # otherwise step over.
    method:   str   = 'RK45'      # 'RK45' | 'DOP853' | 'LSODA' | ...
    rtol:     float = 1e-8
    atol:     float = 1e-10
    max_step: float = 0.5         # cap solver step for time-varying terms


# ---------- Named presets ----------
# Convenience factories that return a ready-to-run SimConfig matching a
# specific worked example from the textbook, so results can be sanity
# checked against a known reference case.

def mc_example_71() -> SimConfig:
    """M&C Example 7.1 — large spacecraft, diagonal J, fast initial tumble."""
    cfg = SimConfig()
    cfg.Ixx, cfg.Iyy, cfg.Izz = 10000.0, 9000.0, 12000.0
    cfg.Ixy = cfg.Ixz = cfg.Iyz = 0.0   # diagonal inertia tensor (no products of inertia)
    cfg.Kp, cfg.Kd = 50.0, 500.0
    cfg.u_max = 100.0
    q = np.array([0.6853, 0.6953, 0.1531, 0.1531])
    cfg.q0 = q / np.linalg.norm(q)      # re-normalize in case the book value isn't exactly unit
    cfg.w0 = np.array([0.53, 0.53, 0.053])      # rad/s -- a fairly aggressive initial tumble
    cfg.t_end = 300.0
    return cfg


def mc_example_72() -> SimConfig:
    """M&C Example 7.2 — smaller spacecraft w/ products of inertia, gentle IC.
    All SimConfig() defaults already match Example 7.2, so this is just an
    alias that makes intent explicit at the call site."""
    return SimConfig()


def spicesat() -> SimConfig:
    """SPICEsat — small-satellite preset.

    Inertia tensor converted from measured values in g*mm^2
    (1 g*mm^2 = 1e-9 kg*m^2). Wheel geometry is unchanged from the
    default 4-wheel NASA pyramid (wheel_tilt_deg); only inertia, gains,
    and torque saturation differ from mc_example_71()/mc_example_72().

    Kp=0.05, Kd=10.0: with Kd raised well above Kp (unusual ratio, but
    needed here), single-axis slews up to ~170 deg and moderate coupled
    multi-axis slews (<=60 deg) converge cleanly. Large SIMULTANEOUS
    multi-axis slews (e.g. 90/90/90) were found to remain unstable (a
    sustained, non-decaying oscillation) even at this Kd -- the same
    gyroscopic/wheel-coupling instability mechanism documented for
    mc_example_72() -- and were not further resolved by raising Kd alone
    (tested up to Kd=100). Single-axis and moderate coupled maneuvers are
    the validated envelope for this preset; large simultaneous multi-axis
    slews are a known open issue, not yet fixed.
    """
    cfg = SimConfig()
    cfg.Ixx, cfg.Iyy, cfg.Izz = 0.09579692958, 0.08815373599, 0.05163644679
    cfg.Ixy, cfg.Ixz, cfg.Iyz = 0.00285635546, 0.00349183595, 0.00514615995
    cfg.u_max = 0.005          # N*m, per-axis wheel/PD torque saturation
    cfg.Kp, cfg.Kd = 0.05, 1.0
    return cfg


# =====================================================================
# Quaternion helpers (scalar-last convention)
# =====================================================================
# All quaternions in this file are ordered q = [q1, q2, q3, q4] with q4 the
# SCALAR part (last, not first — a common source of sign/order bugs when
# cross-referencing other textbooks/libraries that use scalar-first).
# Quaternions represent the body-to-inertial attitude and must stay unit
# norm; solve_ivp only integrates approximately, so simulate() re-normalizes
# q_hist after integration to correct small numerical drift.

def quat_multiply(q, p):
    """q ⊗ p  with scalar-last convention (M&C Eq. 2.82b).

    Quaternion multiplication composes two rotations: if q represents
    "rotate by A" and p represents "rotate by B", q ⊗ p represents
    "rotate by B, then by A" (i.e. p is applied first)."""
    qv, q4 = q[0:3], q[3]
    pv, p4 = p[0:3], p[3]
    vec = q4 * pv + p4 * qv - np.cross(qv, pv)
    sca = q4 * p4 - np.dot(qv, pv)
    return np.hstack((vec, sca))


def quat_inverse(q):
    """Inverse (conjugate) of a unit quaternion, scalar last.

    For a UNIT quaternion the inverse is just the conjugate: negate the
    vector part, keep the scalar part. This represents the opposite
    rotation."""
    return np.hstack((-q[0:3], q[3]))


def quat_kinematics(q, omega):
    """
    Quaternion rate:  q_dot = (1/2) * Omega(omega) * q   (M&C Eq. 2.88).
        q_v_dot = (1/2) ( q4 * omega  -  omega x q_v )
        q_4_dot = -(1/2) omega · q_v

    This is the kinematic differential equation that propagates attitude
    forward in time given the current body rate omega. It has nothing to do
    with forces/torques — that's rigid_body_dynamics() below; this just
    says "how does orientation change given the current spin rate."
    """
    qv, q4 = q[0:3], q[3]
    qv_dot = 0.5 * (q4 * omega - np.cross(omega, qv))
    q4_dot = -0.5 * np.dot(omega, qv)
    return np.hstack((qv_dot, q4_dot))


def euler321_to_quat(roll_deg, pitch_deg, yaw_deg):
    """
    Convert a 3-2-1 (yaw-pitch-roll, Z-Y-X) Euler triple to a scalar-last
    quaternion (M&C App. B.96):  q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll).

    This exists purely as a human-friendly I/O convenience — internally the
    simulation only ever works with quaternions (q0, q_des, etc.), never
    Euler angles, to avoid gimbal-lock singularities during integration.
    Angles in, degrees; used only at scenario-setup time.
    """
    r = np.deg2rad(roll_deg)  / 2.0
    p = np.deg2rad(pitch_deg) / 2.0
    y = np.deg2rad(yaw_deg)   / 2.0
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    q4 = cr * cp * cy + sr * sp * sy
    return np.array([q1, q2, q3, q4])


def quat_to_euler321(q):
    """
    Inverse of `euler321_to_quat`: scalar-last quaternion -> 3-2-1
    (yaw-pitch-roll) Euler angles in RADIANS.  The pitch arg is clipped
    to ±1 to stay numerically safe near gimbal lock.

    Used purely for human-readable plotting/export (e.g. the "roll, pitch,
    yaw" columns in the MATLAB export) — never fed back into the dynamics.
    """
    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
    roll  = np.arctan2(2.0*(q4*q1 + q2*q3),
                       1.0 - 2.0*(q1*q1 + q2*q2))
    sinp  = np.clip(2.0*(q4*q2 - q1*q3), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2.0*(q4*q3 + q1*q2),
                       1.0 - 2.0*(q2*q2 + q3*q3))
    return roll, pitch, yaw


# =====================================================================
# Rigid-body dynamics with reaction wheels
# =====================================================================

def rigid_body_dynamics(omega, J, u_body, h_w_body=None):
    """
    Euler's equation with internal angular momentum (M&C Eq. 7.21):
        J * omega_dot = u_body  -  omega x (J * omega + h_w_body)
    `h_w_body = W @ h_w` is the wheel momentum expressed in the body frame.

    This is the "F = ma" of rotational dynamics: given the net external
    torque u_body and the current spin state, solve for angular
    acceleration. The `omega x (...)` term is the gyroscopic coupling
    between axes that makes rigid-body rotation nonlinear (absent for a
    1-DOF system) — it's why a tumbling body can look chaotic even with zero
    applied torque. `np.linalg.solve` is used instead of an explicit
    inverse of J for better numerical conditioning.
    """
    if h_w_body is None:
        h_w_body = np.zeros(3)
    return np.linalg.solve(J, u_body - np.cross(omega, J @ omega + h_w_body))


def build_wheel_matrix(wheel_tilt_deg):
    """
    Build the 3 x n_w wheel-axis configuration matrix W for the 4-wheel
    NASA pyramid (M&C Sec. 7.2): spin axes tilted `wheel_tilt_deg` from
    +Z, equally spaced 90 deg apart in azimuth.

    Factored out here since it's needed in three places that must all
    agree on the same wheel geometry: simulate() (to run the dynamics),
    _build_trajectory_columns() (to reconstruct u_act for MATLAB export),
    and plot_results() (to reconstruct u_act for the u_act panel).
    """
    beta = np.deg2rad(wheel_tilt_deg)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([[ sb,   0.0, -sb,   0.0],
                     [ 0.0,  sb,   0.0, -sb ],
                     [ cb,   cb,   cb,   cb ]])           # (3 x n_w)


def reconstruct_u_act(tauw_hist, u_hist, W, enable_wheels):
    """
    Reconstruct u_act (body-frame wheel-reaction torque) from a saved
    wheel-torque history, matching the actuator branch in dynamics_rhs:
        wheels ON:  u_act = -W @ tau_w
        wheels OFF: u_act = u_cmd
    tauw_hist, u_hist: (N, n_w) and (N, 3) histories from simulate().
    Used by both plot_results() and _build_trajectory_columns() so the
    two stay consistent with each other.
    """
    if enable_wheels:
        return -(W @ tauw_hist.T).T
    return u_hist


def allocate_wheel_torque(u_cmd, W_pinv):
    """
    Pseudoinverse allocation of a body torque command to per-wheel torques:
        -W @ tau_w = u_cmd     =>     tau_w = -W^+ @ u_cmd

    W is 3x4 (3 body axes, 4 wheels) so this system is under-determined —
    there are infinitely many (tau_w1..tau_w4) combinations that produce the
    same net body torque. The Moore-Penrose pseudoinverse W_pinv picks the
    minimum-norm solution, i.e. the wheel-torque combination that spins the
    wheels up the least for the commanded body torque.
    """
    return -W_pinv @ u_cmd


def wheel_to_ss(num, den):
    """
    Convert a wheel-actuator transfer function tau_actual(s)/tau_cmd(s) =
    num(s)/den(s) into a state-space realization (A, B, C) via
    scipy.signal.tf2ss. Assumes a strictly proper TF (D = 0), true for any
    physically realizable low-pass actuator response.

    Call this ONCE per simulation (in simulate()) -- not per RHS
    evaluation -- and reuse the resulting (A, B, C) via wheel_actuator_dynamics().
    """
    A, B, C, D = tf2ss(num, den)
    if not np.allclose(D, 0.0):
        raise ValueError("wheel_tf_num/wheel_tf_den must be strictly proper (deg(num) < deg(den))")
    return A, B, C


def wheel_actuator_dynamics(x, tau_cmd, A, B, C):
    """
    Per-wheel reaction-wheel actuator dynamics: a low-pass filter between
    the commanded wheel torque and the torque actually delivered, modeling
    finite spin-up/spin-down response instead of an instantaneous actuator.

    (A, B, C) come from wheel_to_ss(cfg.wheel_tf_num, cfg.wheel_tf_den) --
    built ONCE from the transfer function, so changing the actuator's
    response is just editing those two coefficient lists in SimConfig, no
    manual re-derivation of this function.

    x       : (n_states, n_w) filter state, one column per wheel
    tau_cmd : (n_w,) commanded torque per wheel (from allocate_wheel_torque)
    Returns (x_dot, tau_actual): x_dot is (n_states, n_w); tau_actual (n_w,)
    is the filtered torque to actually use in place of tau_cmd.
    """
    x_dot = A @ x + B @ tau_cmd.reshape(1, -1)
    tau_actual = (C @ x).flatten()
    return x_dot, tau_actual


# =====================================================================
# Magnetic field and B-dot momentum-dumping controller
# =====================================================================
# Reaction wheels can only ABSORB momentum, not get rid of it — if the
# spacecraft is under a constant disturbance torque, wheel speeds (and
# stored momentum) will ramp up forever and eventually saturate. The
# magnetorquer dumps that excess momentum by pushing against the Earth's
# magnetic field instead, at the cost of only being able to null the
# momentum component perpendicular to the *current* B vector at any instant.

def magnetic_field_body(t):
    """
    Toy Earth B-field in the body frame, ~30 µT magnitude, rotating with
    one 90-minute LEO orbital period.  Replace with IGRF + attitude
    rotation for a high-fidelity simulation.

    This is deliberately simplified: it ignores the spacecraft's actual
    attitude and orbital geometry, and just rotates a fixed-magnitude
    vector at the orbital rate to emulate the fact that, over one orbit,
    the field direction sweeps through the body frame — which is what lets
    the B-dot law eventually null momentum on ALL three axes, not just one.
    """
    B0 = 30e-6                              # Tesla
    omega_orb = 2.0 * np.pi / 5400.0        # rad/s
    return B0 * np.array([np.cos(omega_orb * t),
                          0.3,
                          np.sin(omega_orb * t)])


def bdot_momentum_dump(h_w_body, B_body, K_dump, m_max):
    """
    Cross-product ("B-dot-style") momentum-dumping law, M&C Sec. 7.5.
        m       = K_dump * (h_w_body x B) / |B|^2      (commanded dipole)
        tau_mag = m x B = -K_dump * h_w_perp_to_B      (resulting torque)
    Output dipole is clipped element-wise to [-m_max, +m_max].

    The resulting torque tau_mag = m x B only ever opposes the component of
    wheel momentum h_w_body that is PERPENDICULAR to B — the component
    parallel to B can't be touched by a magnetic dipole no matter how it's
    commanded (cross product of parallel vectors is zero). That's fine here
    because B_body rotates over the orbit (see magnetic_field_body), so the
    "un-reachable" axis keeps changing and average momentum still trends to
    zero over time.
    """
    B2 = float(np.dot(B_body, B_body))
    if B2 < 1e-14:                          # avoid divide-by-zero
        return np.zeros(3)
    m_cmd = K_dump * np.cross(h_w_body, B_body) / B2
    return np.clip(m_cmd, -m_max, m_max)


# =====================================================================
# Propellant slosh model
# =====================================================================
# Liquid propellant sloshing inside partially-filled tanks behaves like a
# lightly-damped spring-mass oscillator that's excited by the spacecraft's
# own rotation. We don't simulate the fluid directly (no CFD) — instead each
# axis's slosh torque Ts is treated as a STATE driven by a forced 2nd-order
# ODE, coupled to the rigid-body dynamics through omega/omega_dot going in
# and Ts feeding back out as an extra disturbance torque (see dynamics_rhs
# step 4: `slosh_torque = Ts`). Two alternate ODE forms are provided below;
# exactly one is active per run, chosen by cfg.slosh_model.

def slosh_coefficient(t, nom, amp, omega):
    """
    Time-varying scalar coefficient with a DC offset:
        f(t) = nom + amp * sin(omega * t)
    Keep nom > |amp| so K_s, C_s never go negative (which would be
    unphysical — negative stiffness or damping).

    Used by slosh_dynamics_bourdelle() to let each of A_s/B_s/C_s/K_s
    optionally wander sinusoidally in time (e.g. to emulate a slowly
    draining tank changing the slosh mode's characteristics); with amp=0 it
    just returns the constant 'nom'.
    """
    return nom + amp * np.sin(omega * t)


def slosh_dynamics_bourdelle(omega_body, omega_dot, Ts, Ts_dot, t, slosh_params):
    """
    Second-order propellant-slosh model driven by spacecraft body motion:
        Ts_ddot = -A_s(t)*omega - B_s(t)*omega_dot
                  - C_s(t)*Ts_dot - K_s(t)*Ts
    Returns Ts_ddot (3,).  `slosh_params` is a dict keyed by 'A','B','C','K',
    each holding {'nom':..., 'amp':..., 'omega':...}.

    Physically: K_s is the slosh mode's "spring stiffness" (restoring term
    pulling Ts back to zero), C_s is its damping, and A_s/B_s are forcing
    gains that convert the spacecraft's own body rate/acceleration into a
    slosh-torque forcing term — i.e. the more aggressively the spacecraft
    rotates or accelerates, the harder the propellant sloshes back.
    This is the ORIGINAL slosh model (renamed from `slosh_dynamics`);
    coefficients can each independently vary sinusoidally in time via
    slosh_coefficient() above.
    """
    A = slosh_coefficient(t, **slosh_params['A'])
    B = slosh_coefficient(t, **slosh_params['B'])
    C = slosh_coefficient(t, **slosh_params['C'])
    K = slosh_coefficient(t, **slosh_params['K'])
    return -A * omega_body - B * omega_dot - C * Ts_dot - K * Ts


def slosh_dynamics_testing(omega_body, omega_dot, Ts, Ts_dot, testing_params):
    """
    Alternate second-order propellant-slosh model with rate-dependent
    stiffness/damping saturation:
        Ts_ddot = -omega_n^2 * Ts
                  - 2*zeta*omega_n / (1 + (Omega/Omega_max)^2) * Ts_dot
                  + omega_n^2 * (1 + (Omega/Omega_max)^2) * Omega_dot
    where Omega = omega_body (per axis) and Omega_dot = omega_dot (per axis).
    `testing_params` holds {'omega_n':..., 'zeta':..., 'omega_max':...}.

    Unlike the 'bourdelle' model (constant/sinusoidal coefficients), here
    the effective damping and forcing gain both scale with
    (1 + (Omega/Omega_max)^2): as the body spin rate Omega approaches the
    max expected rotation rate Omega_max, damping on the Ts_dot term
    SOFTENS (denominator grows) while the forcing from Omega_dot STRENGTHENS
    (same factor as numerator) — modeling a slosh mode that becomes more
    lightly damped and more strongly excited at high spin rates. K = omega_n^2
    and C = 2*zeta*omega_n follow the standard 2nd-order-system relations
    between natural frequency/damping ratio and stiffness/damping.
    """
    omega_n   = testing_params['omega_n']
    zeta      = testing_params['zeta']
    omega_max = testing_params['omega_max']

    K = omega_n ** 2
    ratio_sq = (omega_body / omega_max) ** 2   # (Omega / Omega_max)^2, per axis

    return (-K * Ts                                                  # stiffness (restoring) term
            - (2.0 * zeta * omega_n / (1.0 + ratio_sq)) * Ts_dot     # rate-softened damping term
            + K * (1.0 + ratio_sq) * omega_dot)                      # rate-amplified forcing term


# =====================================================================
# Quaternion PD attitude controller
# =====================================================================

def pd_controller(q, omega, q_des, Kp, Kd, u_max=np.inf):
    """
    Quaternion-feedback PD law (M&C Sec. 7.4), with per-axis saturation:
        u = clip( -Kp * sign(dq4) * dq_vec  -  Kd * omega,  ±u_max )
    where dq = q_des^(-1) ⊗ q.  The sign(dq4) factor selects the shortest
    rotation (prevents 'unwinding').  Returns (u, dq).

    dq is the "error quaternion" — the rotation still needed to go from the
    current attitude q to the desired attitude q_des. Its vector part
    dq_vec is (for small errors) proportional to the axis-angle rotation
    error, so -Kp*dq_vec acts like a proportional spring pulling the
    spacecraft toward q_des, while -Kd*omega is a damping term. Because a
    quaternion and its negative represent the SAME physical attitude, dq4
    can flip sign between calls even though the physical error is
    continuous — without the sign(dq4) correction, the controller could
    command a rotation the "long way around" (> 180 deg) purely due to that
    sign ambiguity; this is the classic "unwinding" bug in quaternion
    control.
    """
    dq = quat_multiply(quat_inverse(q_des), q)
    dq_vec, dq4 = dq[0:3], dq[3]
    u_unsat = -Kp * np.sign(dq4) * dq_vec - Kd * omega
    u = np.clip(u_unsat, -u_max, u_max)
    return u, dq


# =====================================================================
# State packing for solve_ivp
#   y = [ q (4) ; omega (3) ; h_w (n_w) ; Ts (3) ; Ts_dot (3) ]
#   total size = 13 + n_w   (= 17 for the 4-wheel default)
# =====================================================================

def pack_state(q, w, h_w, Ts, Tsd, x_w=None):
    """Flatten the individual state pieces into the single 1-D vector y
    that scipy.integrate.solve_ivp requires (it only knows how to integrate
    a flat array, not a structured collection of named quantities).

    x_w is the (n_states, n_w) reaction-wheel actuator filter state
    (see wheel_actuator_dynamics), flattened. Omit it (or pass None) when
    wheel actuator dynamics are disabled -- it contributes zero extra
    entries in that case, so the state vector size adjusts automatically."""
    x_w_flat = np.zeros(0) if x_w is None else np.asarray(x_w).flatten()
    return np.concatenate([q, w, h_w, Ts, Tsd, x_w_flat])


def unpack_state(y, n_w, n_wheel_states=0):
    """Inverse of pack_state(): slice the flat solver state vector y back
    into its named physical quantities. n_w (number of wheels) is needed
    because it determines where the fixed-size Ts/Tsd blocks start;
    n_wheel_states is the reaction-wheel actuator filter's order (0 when
    wheel actuator dynamics are disabled, matching pack_state's default).

    Returns (q, w, h_w, Ts, Tsd, x_w), where x_w is (n_wheel_states, n_w)."""
    q   = y[0:4]
    w   = y[4:7]
    h_w = y[7:7 + n_w]
    Ts  = y[7 + n_w : 10 + n_w]
    Tsd = y[10 + n_w : 13 + n_w]
    x_w_flat = y[13 + n_w : 13 + n_w + n_wheel_states * n_w]
    x_w = x_w_flat.reshape(n_wheel_states, n_w)
    return q, w, h_w, Ts, Tsd, x_w


# =====================================================================
# ODE right-hand side for solve_ivp
# =====================================================================

def dynamics_rhs(t, y, params):
    """
    Compute dy/dt for the full coupled system.  All controllers (PD,
    magnetorquer, wheel allocator) are evaluated continuously here as
    functions of the current state — i.e. the closed-loop is treated as a
    continuous-time system.

    Because Ts (slosh torque) is a STATE that appears in Euler's equation,
    and Ts_ddot consumes the just-computed omega_dot, there is no algebraic
    loop: we compute omega_dot first using the current Ts, then evaluate
    the slosh second derivative.
    """
    # Unpack the flat solver state vector into named physical quantities.
    q, w, h_w, Ts, Tsd, x_w = unpack_state(y, params['n_w'], params['n_wheel_states'])
    W      = params['W']       # 3 x n_w wheel-axis mapping matrix
    W_pinv = params['W_pinv']  # its pseudoinverse, for torque allocation

    # 1) Magnetorquer (toggle). Computes the momentum-dumping torque from
    #    the CURRENT wheel momentum and the CURRENT (time-varying) B-field.
    #    When disabled, this subsystem contributes nothing.
    if params['enable_magnetorquer']:
        B_body  = magnetic_field_body(t)
        m_cmd   = bdot_momentum_dump(W @ h_w, B_body,
                                     params['K_dump'], params['m_max'])
        tau_mag = np.cross(m_cmd, B_body)
    else:
        tau_mag = np.zeros(3)

    # 2) PD attitude controller -> desired body torque. This runs
    #    unconditionally (it's always "on"); what varies below is how its
    #    output gets applied to the body.
    u_cmd, _ = pd_controller(q, w, params['q_des'],
                             params['Kp'], params['Kd'], params['u_max'])

    # 3) Actuator branch: reaction wheels (toggle) or ideal torque actuator.
    #    With wheels OFF, the PD output drives the body directly and the wheel
    #    state is frozen (no momentum accumulation, no reaction torque).
    if params['enable_wheels']:
        tau_w_cmd = allocate_wheel_torque(u_cmd, W_pinv)   # per-wheel torque commands
        if params['enable_wheel_dynamics']:
            # Wheels don't respond instantly -- run the commanded torque
            # through the actuator's low-pass filter (see
            # wheel_actuator_dynamics) to get the torque actually delivered.
            x_w_dot, tau_w = wheel_actuator_dynamics(
                x_w, tau_w_cmd, params['wheel_A'], params['wheel_B'], params['wheel_C'])
        else:
            tau_w   = tau_w_cmd            # instantaneous actuator (as before)
            x_w_dot = np.zeros((params['n_wheel_states'], params['n_w']))
        u_act  = -W @ tau_w                 # wheel reaction on the body (Newton's 3rd law)
        h_body = W @ h_w                    # wheel momentum in body frame
    else:
        tau_w   = np.zeros(params['n_w'])
        x_w_dot = np.zeros((params['n_wheel_states'], params['n_w']))
        u_act  = u_cmd                      # ideal body-torque actuator
        h_body = np.zeros(3)

    # 4) Slosh torque on the body (zero if disabled). Ts is itself one of
    #    the integrated states (see step 7), so this just reads its current
    #    value — it doesn't compute anything new here.
    slosh_torque = Ts if params['enable_slosh'] else np.zeros(3)

    # 5) Euler's equation -> body angular acceleration. This is the moment
    #    ALL torque sources (actuator + magnetorquer + slosh disturbance)
    #    combine into the single net torque that determines w_dot.
    u_body = u_act + tau_mag + slosh_torque
    w_dot  = rigid_body_dynamics(w, params['J'], u_body, h_body)

    # 6) Quaternion kinematics and wheel-momentum law. q_dot integrates
    #    attitude forward given the current spin rate w; h_dot integrates
    #    wheel momentum forward given the commanded wheel torques.
    q_dot = quat_kinematics(q, w)
    h_dot = tau_w                          # zero when wheels disabled; the
                                            # ACTUAL (post-filter) torque, so
                                            # wheel spin-up matches what's
                                            # really delivered to the body

    # 7) Slosh second-order ODE (uses w_dot just computed in step 5).
    #    NOTE ON CAUSALITY: w_dot depends on Ts (a STATE, already known at
    #    the start of this call), NOT on Ts_ddot -- so there's no circular
    #    dependency here even though slosh feeds back into the rigid-body
    #    torque balance. Exactly one of the two interchangeable slosh models
    #    is evaluated, selected by params['slosh_model'].
    if params['enable_slosh']:
        Ts_dot = Tsd    # first slosh state's derivative is just the second slosh state
        if params['slosh_model'] == 'testing':
            Tsd_dot = slosh_dynamics_testing(w, w_dot, Ts, Tsd,
                                             params['slosh_testing_params'])
        else:
            Tsd_dot = slosh_dynamics_bourdelle(w, w_dot, Ts, Tsd, t,
                                               params['slosh_params'])
    else:
        Ts_dot  = np.zeros(3)
        Tsd_dot = np.zeros(3)

    # Re-flatten everything back into a single vector for solve_ivp.
    return np.concatenate([q_dot, w_dot, h_dot, Ts_dot, Tsd_dot, x_w_dot.flatten()])


# =====================================================================
# Simulation driver
# =====================================================================

def simulate(cfg: SimConfig = None):
    """
    Run the closed-loop simulation with scipy.integrate.solve_ivp.

    Parameters
    ----------
    cfg : SimConfig (or None — defaults to SimConfig())
        All scenario knobs.  Build with a preset (e.g. mc_example_71()) and
        override individual fields before passing in.
    """
    if cfg is None:
        cfg = SimConfig()

    # ---- Build the inertia tensor from independent components ----
    # Note the sign flip on the off-diagonal products of inertia: SimConfig
    # stores Ixy/Ixz/Iyz as positive magnitudes (M&C Eq. 3.13 convention),
    # but the actual tensor entries are negative. Getting this sign wrong is
    # a classic bug, hence the explicit symmetry + positive-definiteness
    # assertions right below -- they'll immediately flag a malformed J
    # rather than silently producing wrong (but not obviously broken) dynamics.
    J = np.array([[ cfg.Ixx, -cfg.Ixy, -cfg.Ixz],
                  [-cfg.Ixy,  cfg.Iyy, -cfg.Iyz],
                  [-cfg.Ixz, -cfg.Iyz,  cfg.Izz]])
    assert np.allclose(J, J.T), "Inertia tensor must be symmetric"
    assert np.all(np.linalg.eigvalsh(J) > 0), "Inertia tensor must be positive-definite"

    # ---- Reaction-wheel configuration: 4-wheel NASA pyramid (M&C Sec. 7.2) ----
    W = build_wheel_matrix(cfg.wheel_tilt_deg)
    n_w = W.shape[1]
    W_pinv = np.linalg.pinv(W)

    # ---- Reaction-wheel actuator dynamics (optional low-pass filter) ----
    # Built ONCE here from the transfer function coefficients (not per RHS
    # evaluation) -- see wheel_to_ss()/wheel_actuator_dynamics(). When
    # disabled, n_wheel_states=0 so the filter contributes no extra states
    # and dynamics_rhs falls back to the instantaneous actuator.
    if cfg.enable_wheel_dynamics:
        wheel_A, wheel_B, wheel_C = wheel_to_ss(cfg.wheel_tf_num, cfg.wheel_tf_den)
        n_wheel_states = wheel_A.shape[0]
    else:
        wheel_A = wheel_B = wheel_C = None
        n_wheel_states = 0

    # ---- Initial state vector ----
    # .copy() everywhere so solve_ivp's internal state array never aliases
    # (and thus can never accidentally mutate) the SimConfig's own arrays --
    # important since run_batch() reuses one base_cfg's arrays by reference
    # across many scenario copies.
    y0 = pack_state(cfg.q0.copy(),
                    cfg.w0.copy(),
                    np.zeros(n_w),           # wheels always start at rest
                    cfg.Ts0.copy(),
                    cfg.Tsd0.copy(),
                    np.zeros((n_wheel_states, n_w)))   # wheel filter starts at rest

    # ---- Output time grid ----
    # This is ONLY the set of times at which solve_ivp reports back a
    # solution sample (dt_eval) -- it does not control the adaptive
    # integrator's internal step size, which is chosen independently
    # subject to rtol/atol/max_step below.
    t_eval = np.arange(0.0, cfg.t_end + cfg.dt_eval, cfg.dt_eval)

    # ---- Parameter bundle for the RHS closure ----
    # Everything dynamics_rhs() needs, gathered into one dict since
    # solve_ivp's callback signature only accepts (t, y) plus this via the
    # lambda closure below -- keeps dynamics_rhs a pure function of
    # (t, y, params) rather than reaching into globals or cfg directly.
    params = {
        'J': J, 'W': W, 'W_pinv': W_pinv, 'n_w': n_w,
        'Kp': cfg.Kp, 'Kd': cfg.Kd, 'u_max': cfg.u_max, 'q_des': cfg.q_des,
        'K_dump': cfg.K_dump, 'm_max': cfg.m_max,
        'slosh_model': cfg.slosh_model,
        'slosh_params': cfg.slosh_params,
        'slosh_testing_params': cfg.slosh_testing_params,
        'enable_slosh': cfg.enable_slosh,
        'enable_magnetorquer': cfg.enable_magnetorquer,
        'enable_wheels': cfg.enable_wheels,
        'enable_wheel_dynamics': cfg.enable_wheel_dynamics,
        'wheel_A': wheel_A, 'wheel_B': wheel_B, 'wheel_C': wheel_C,
        'n_wheel_states': n_wheel_states,
    }

    # ---- Integrate ----
    sol = solve_ivp(
        fun=lambda tt, yy: dynamics_rhs(tt, yy, params),
        t_span=(0.0, cfg.t_end), y0=y0,
        method=cfg.method, t_eval=t_eval,
        rtol=cfg.rtol, atol=cfg.atol, max_step=cfg.max_step,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    print(f"solve_ivp: {sol.nfev} RHS evals, {sol.t.size} output samples")

    t = sol.t
    Y = sol.y.T            # (N, 13 + n_w + n_wheel_states*n_w) -- rows are timesteps
    N = len(t)

    # ---- Extract state trajectories ----
    # Same slicing convention as unpack_state(), just applied to the whole
    # (N, ...) history array at once instead of one instantaneous state.
    q_hist  = Y[:, 0:4]
    w_hist  = Y[:, 4:7]
    hw_hist = Y[:, 7:7 + n_w]
    Ts_hist = Y[:, 7 + n_w : 10 + n_w]
    # Wheel actuator filter state history, (N, n_wheel_states, n_w); empty
    # (n_wheel_states=0) when wheel actuator dynamics are disabled.
    xw_hist = Y[:, 13 + n_w:].reshape(N, n_wheel_states, n_w)
    # Re-normalize quaternions (solve_ivp drift is small but nonzero).
    q_hist = q_hist / np.linalg.norm(q_hist, axis=1, keepdims=True)

    # ---- Reconstruct controller telemetry at each sample ----
    # solve_ivp only integrates the STATE (q, w, h_w, Ts, Tsd); intermediate
    # quantities computed inside dynamics_rhs (like u_cmd, tau_mag, the
    # B-field) are NOT saved automatically. To get them for plotting/export,
    # we re-run the same per-step calculations here, after the fact, once
    # per saved sample -- cheap relative to the integration itself since
    # there's no ODE solving involved, just direct function evaluation.
    dq_hist   = np.zeros((N, 4))
    u_hist    = np.zeros((N, 3))
    tauw_hist = np.zeros((N, n_w))
    m_hist    = np.zeros((N, 3))
    tmag_hist = np.zeros((N, 3))
    B_hist    = np.zeros((N, 3))

    for k in range(N):
        q, w, h_w = q_hist[k], w_hist[k], hw_hist[k]
        if cfg.enable_magnetorquer:
            B_body  = magnetic_field_body(t[k])
            m_cmd   = bdot_momentum_dump(W @ h_w, B_body, cfg.K_dump, cfg.m_max)
            tau_mag = np.cross(m_cmd, B_body)
        else:
            B_body, m_cmd, tau_mag = np.zeros(3), np.zeros(3), np.zeros(3)
        u_cmd, dq = pd_controller(q, w, cfg.q_des, cfg.Kp, cfg.Kd, cfg.u_max)
        if not cfg.enable_wheels:
            tau_w = np.zeros(n_w)
        elif cfg.enable_wheel_dynamics:
            # Use the ACTUAL (post-filter) torque consistent with what was
            # really integrated, not the instantaneous command -- read from
            # the wheel filter's own state/output rather than recomputing
            # allocate_wheel_torque() alone (which would just give the
            # unfiltered command and misrepresent what the wheel delivered).
            tau_w = (wheel_C @ xw_hist[k]).flatten()
        else:
            tau_w = allocate_wheel_torque(u_cmd, W_pinv)

        dq_hist[k]   = dq
        u_hist[k]    = u_cmd
        tauw_hist[k] = tau_w
        m_hist[k]    = m_cmd
        tmag_hist[k] = tau_mag
        B_hist[k]    = B_body

    return (t, q_hist, dq_hist, w_hist, u_hist,
            hw_hist, tauw_hist, m_hist, tmag_hist, B_hist, Ts_hist)


# =====================================================================
# Plotting
# =====================================================================

def plot_results(t, q_hist, dq_hist, w_hist, u_hist,
                 hw_hist, tauw_hist, m_hist, tmag_hist, B_hist, Ts_hist,
                 cfg=None):
    """Main 8-panel telemetry figure: states + controller outputs + slosh.

    Takes the full `results` tuple from simulate() unpacked as positional
    args (call as `plot_results(*results, cfg=cfg)`); each panel is one
    physical quantity's time history across all 3 (or n_w, for wheels)
    components.

    cfg (the SimConfig used to produce `results`) is required to
    reconstruct u_act -- see build_wheel_matrix() -- since u_act depends
    on wheel geometry and the enable_wheels toggle, neither of which is
    part of the plain results tuple."""
    if cfg is None:
        raise ValueError("plot_results() requires cfg (the SimConfig used "
                         "to produce `results`) to reconstruct u_act")

    fig, ax = plt.subplots(8, 1, figsize=(10, 16), sharex=True)

    ax[0].plot(t, q_hist)
    ax[0].set_ylabel("Quaternion")
    ax[0].legend(["q1", "q2", "q3", "q4 (scalar)"], loc="right")
    ax[0].grid(True)

    ax[1].plot(t, dq_hist)
    ax[1].set_ylabel("Error quaternion")
    ax[1].legend(["dq1", "dq2", "dq3", "dq4 (scalar)"], loc="right")
    ax[1].grid(True)

    N = len(t)
    eul_deg = np.array([np.rad2deg(quat_to_euler321(q)) for q in q_hist])
    ax[2].plot(t, eul_deg)
    ax[2].set_ylabel("Euler angle (deg)")
    ax[2].legend(["roll  ($\\phi$)", "pitch ($\\theta$)", "yaw  ($\\psi$)"], loc="right")
    ax[2].grid(True)

    ax[3].plot(t, w_hist)
    ax[3].set_ylabel("Body rate (rad/s)")
    ax[3].legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"], loc="right")
    ax[3].grid(True)

    # u_act: the body-frame wheel-reaction torque alone (NOT combined with
    # tau_mag -- that combined quantity is a separate thing, only used for
    # the MATLAB export's ux,uy,uz columns; see attitude_dynamics_model.tex
    # Sec. "Torque Signal Chain").
    W = build_wheel_matrix(cfg.wheel_tilt_deg)
    u_act = reconstruct_u_act(tauw_hist, u_hist, W, cfg.enable_wheels)
    ax[4].plot(t, u_act)
    ax[4].set_ylabel(r"$u_{act}$ (N·m)")
    ax[4].legend([r"$u_{act,x}$", r"$u_{act,y}$", r"$u_{act,z}$"], loc="right")
    ax[4].grid(True)

    n_w = hw_hist.shape[1]
    wheel_labels = [f"wheel {i+1}" for i in range(n_w)]

    ax[5].plot(t, hw_hist)
    ax[5].set_ylabel("Wheel momentum (N·m·s)")
    ax[5].legend(wheel_labels, loc="right")
    ax[5].grid(True)

    ax[6].plot(t, tmag_hist)
    ax[6].set_ylabel("Mag. torque (N·m)")
    ax[6].legend([r"$\tau_{mag,x}$", r"$\tau_{mag,y}$", r"$\tau_{mag,z}$"], loc="right")
    ax[6].grid(True)

    ax[7].plot(t, Ts_hist)
    ax[7].set_ylabel("Slosh torque $T_s$ (N·m)")
    ax[7].set_xlabel("Time (s)")
    ax[7].legend([r"$T_{s,x}$", r"$T_{s,y}$", r"$T_{s,z}$"], loc="right")
    ax[7].grid(True)

    plt.tight_layout()


def _scenario_start_target_deg(sc, cfg):
    """
    Return (q0_deg, qdes_deg) — human-readable [roll, pitch, yaw] degree
    triples describing a scenario's start/target attitude.

    Prefers explicit 'q0_deg'/'qdes_deg' entries in the scenario dict (the
    EXACT degrees the caller originally specified) over round-tripping the
    quaternion through quat_to_euler321(). That round trip is lossy at
    gimbal lock (pitch = +-90 deg): infinitely many (roll, pitch, yaw)
    triples map to the same quaternion there, so quat_to_euler321() can
    report a technically-equivalent but visually different triple (e.g.
    [180, 90, 180] instead of the [0, 90, 0] that was actually requested).
    Falls back to that round trip only when the scenario didn't record its
    original degrees.
    """
    if 'q0_deg' in sc:
        q0_deg = np.asarray(sc['q0_deg'], dtype=float)
    else:
        q0_deg = np.rad2deg(quat_to_euler321(sc.get('q0', cfg.q0)))

    if 'qdes_deg' in sc:
        qdes_deg = np.asarray(sc['qdes_deg'], dtype=float)
    else:
        qdes_deg = np.rad2deg(quat_to_euler321(sc.get('q_des', cfg.q_des)))

    return q0_deg, qdes_deg


def plot_batch(batch_results, scenarios=None, cfg=None, save_dir=None):
    """
    Plot each trajectory from a `run_batch()` result in its own separate
    figure window, labeled with its scenario index and, if available, its
    start/target Euler angles.

    For each scenario this opens the 8-panel `plot_results` figure
    (quaternion, error quaternion, Euler angles, body rate, u_act, wheel
    momentum, magnetorquer torque, slosh torque).

    Parameters
    ----------
    batch_results : list of results tuples, as returned by run_batch().
    scenarios : list of dict, optional
        The same scenario list passed to run_batch(); if given, each
        figure's title includes that scenario's start/target angles (see
        `_scenario_start_target_deg` for how those degrees are obtained).
    cfg : SimConfig, optional
        The base_cfg used for the batch; only needed as a fallback source
        of q0/q_des if a scenario doesn't override them.
    save_dir : str, optional
        If given, save each figure as a PNG in this directory (created if
        it doesn't already exist), named `scenario{i:02d}_telemetry.png`.
        Figures still open as normal windows either way -- this only
        additionally writes them to disk.
    """
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)

    for i, results in enumerate(batch_results):
        title = f"Scenario {i}"
        if scenarios is not None:
            sc = scenarios[i]
            q0_deg, qdes_deg = _scenario_start_target_deg(sc, cfg or SimConfig())
            title += f"  start={np.round(q0_deg, 1)}  target={np.round(qdes_deg, 1)}"

        # 8-panel telemetry figure (states + controller outputs + slosh).
        plot_results(*results, cfg=cfg)
        fig = plt.gcf()
        fig.suptitle(title)
        # plot_results() calls plt.tight_layout() internally, BEFORE this
        # suptitle is added -- so without reserving extra headroom here,
        # the title can crowd or overlap the topmost subplot.
        fig.subplots_adjust(top=0.94)
        fig.canvas.manager.set_window_title(title + " (telemetry)")
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f"scenario{i:02d}_telemetry.png"),
                       dpi=150, bbox_inches="tight")

    if save_dir is not None:
        print(f"Saved {len(batch_results)} figures to {save_dir}")


# =====================================================================
# Export to MATLAB (.mat)
# =====================================================================

def _build_trajectory_columns(results, cfg):
    """
    Build the (nt, 19) [time + 18 data-column] matrix and column labels for
    one trajectory's `results` tuple (as returned by `simulate()`).

    Columns: t, angles(3), omega(3), omega_dot(3), control_torque(3),
    Ts(3), Fs(3).

    control_torque = u_act + tau_mag (the actuator + magnetorquer torque
    actually applied to the body), excluding the slosh disturbance torque Ts
    (see dynamics_rhs, u_body = u_act + tau_mag + slosh_torque).

    Fs = [Fsx, Fsy, Fsz] is a placeholder for sloshing forces — not yet
    modelled, so held at zero here until that physics is added.

    omega_dot is not stored by `simulate()`, so it is reconstructed here via
    numerical differentiation (np.gradient) of the saved omega history
    rather than by re-running the RHS.
    """
    (t, q_hist, dq_hist, w_hist, u_hist,
     hw_hist, tauw_hist, m_hist, tmag_hist, B_hist, Ts_hist) = results

    N = len(t)
    angles = np.array([quat_to_euler321(q_hist[k]) for k in range(N)])
    omega_dot = np.gradient(w_hist, t, axis=0)
    Fs = np.zeros((N, 3))

    # Reconstruct u_act, matching the actuator branch in dynamics_rhs.
    W = build_wheel_matrix(cfg.wheel_tilt_deg)
    u_act = reconstruct_u_act(tauw_hist, u_hist, W, cfg.enable_wheels)
    control_torque = u_act + tmag_hist

    data = np.column_stack([t, angles, w_hist, omega_dot, control_torque, Ts_hist, Fs])
    labels = ["t",
              "roll", "pitch", "yaw",
              "wx", "wy", "wz",
              "wx_dot", "wy_dot", "wz_dot",
              "ux", "uy", "uz",
              "Tsx", "Tsy", "Tsz",
              "Fsx", "Fsy", "Fsz"]
    return data, labels


def export_to_mat(results, cfg, filepath="sim_results.mat"):
    """
    Export a single trajectory's telemetry to a MATLAB .mat file.
    See `_build_trajectory_columns` for the column layout.
    """
    from scipy.io import savemat

    data, labels = _build_trajectory_columns(results, cfg)
    savemat(filepath, {"data": data, "labels": labels})
    print(f"Saved {data.shape[0]} samples x {data.shape[1]} columns to {filepath}")


# =====================================================================
# Multi-trajectory batch support
# =====================================================================

def run_batch(base_cfg: SimConfig, scenarios: list) -> list:
    """
    Run one simulation per scenario, reusing `base_cfg` (fixed inertia,
    wheel geometry, gains, slosh model) for everything except the
    per-trajectory overrides.

    Parameters
    ----------
    base_cfg : SimConfig
        The shared satellite/controller configuration.
    scenarios : list of dict
        Each dict may set 'q0', 'q_des', and/or 'w0' (numpy arrays) to
        override the corresponding field on a fresh copy of base_cfg.

    Returns
    -------
    list of results tuples, one per scenario, in the same order.
    """
    all_results = []
    for i, sc in enumerate(scenarios):
        # deepcopy is essential here, not just a copy() -- SimConfig holds
        # nested mutable containers (numpy arrays, the slosh_params dict),
        # and a shallow copy would leave those SHARED across scenarios, so
        # overriding cfg.q0 below could silently mutate base_cfg's array
        # (or a previous iteration's cfg) instead of creating an independent one.
        cfg = copy.deepcopy(base_cfg)
        if 'q0' in sc:
            cfg.q0 = sc['q0']
        if 'q_des' in sc:
            cfg.q_des = sc['q_des']
        if 'w0' in sc:
            cfg.w0 = sc['w0']
        print(f"--- Trajectory {i + 1}/{len(scenarios)} ---")
        all_results.append(simulate(cfg))
    return all_results


def export_batch_to_mat(batch_results, scenarios, cfg, filepath="sim_results_batch.mat"):
    """
    Export all trajectories from `run_batch` to a single MATLAB .mat file.

    Saves `trajectories`: a 1xN cell array (N = number of scenarios), each
    cell holding a struct with fields:
        data    - (nt, 19) matrix, see `_build_trajectory_columns`
        labels  - 1x19 cell array of column names
        q0_deg  - [roll, pitch, yaw] initial Euler angles (deg)
        qdes_deg- [roll, pitch, yaw] desired Euler angles (deg)
    """
    from scipy.io import savemat

    traj_structs = []
    for results, sc in zip(batch_results, scenarios):
        data, labels = _build_trajectory_columns(results, cfg)
        # Record each scenario's actual start/target angles (in human-
        # readable degrees) alongside its data, so MATLAB-side code can
        # identify which trajectory is which without re-deriving it from q0.
        # See _scenario_start_target_deg for why this prefers the
        # scenario's own recorded degrees over quat_to_euler321 (gimbal
        # lock at pitch = +-90 deg makes that round trip lossy/ambiguous).
        q0_deg, qdes_deg = _scenario_start_target_deg(sc, cfg)
        traj_structs.append({
            "data": data,
            "labels": labels,
            "q0_deg": q0_deg,
            "qdes_deg": qdes_deg,
        })

    # scipy.io.savemat turns a numpy object array of dicts into a MATLAB
    # cell array of structs -- this is what gives MATLAB the
    # `trajectories{i}.data` / `.labels` / etc. access pattern.
    trajectories = np.empty((1, len(traj_structs)), dtype=object)
    for i, ts in enumerate(traj_structs):
        trajectories[0, i] = ts

    savemat(filepath, {"trajectories": trajectories})
    print(f"Saved {len(traj_structs)} trajectories to {filepath}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    # This block only runs when the file is executed directly (e.g. `python
    # rigid_body_pd_v3.py` or Spyder's Run), NOT when it's imported as a
    # module (e.g. `from rigid_body_pd_v3 import simulate` in a test file)
    # -- so importing this file for its functions never has the side effect
    # of popping up plot windows.

    # ---- Pick a preset, then tweak any field you want before running ----
    # Available presets:
    #     SimConfig()         # baseline (= M&C Example 7.2)
    #     mc_example_71()
    #     mc_example_72()
    #     cfg = mc_example_72()
    cfg = spicesat()
    # Examples of in-place tweaks:
    cfg.enable_slosh = True
    # cfg.enable_magnetorquer = True
    # cfg.enable_wheels = False
    # cfg.Kp, cfg.Kd = 15.0, 200.0
    # cfg.t_end = 600.0

    results = simulate(cfg)
    plot_results(*results, cfg=cfg)
    plt.show()
