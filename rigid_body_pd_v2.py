"""
Rigid-body attitude dynamics with quaternion PD control, reaction wheels,
magnetic momentum dumping, and a propellant-slosh disturbance torque.

Version 2 — refactored so that all spacecraft parameters, gains, initial
conditions, time settings, and toggles live in a single SimConfig dataclass
at the top of the file.  Named preset factory functions are provided for the
worked examples in Markley & Crassidis (Examples 7.1 and 7.2).

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

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp


# =====================================================================
# Configuration  ── all knobs in one place; edit here to change the sim
# =====================================================================

@dataclass
class SimConfig:
    """All satellite parameters, gains, ICs, time, and toggles."""

    # ---- Inertia tensor components (M&C Eq. 3.13), kg·m^2 ----
    Ixx: float = 6400.0
    Iyy: float = 4730.0
    Izz: float = 8160.0
    Ixy: float = 76.4        # products of inertia (signs handled in J builder)
    Ixz: float = 25.6
    Iyz: float = 40.0

    # ---- Reaction-wheel geometry (4-wheel NASA pyramid) ----
    wheel_tilt_deg: float = 54.7356      # acos(1/sqrt(3))

    # ---- Control gains and saturation ----
    Kp:     float = 10.0     # proportional gain
    Kd:     float = 150.0    # derivative gain
    u_max:  float = 1000.0   # N·m,    per-axis PD torque saturation
    K_dump: float = 1.0      # 1/s,    magnetic momentum-dump gain
    m_max:  float = 500.0    # A·m^2,  per-axis magnetorquer dipole saturation

    # ---- Slosh model selector: 'bourdelle' (default) or 'testing' ----
#    slosh_model: str = 'bourdelle'
    slosh_model: str = 'testing'


    # ---- Slosh model (bourdelle):  f(t) = nom + amp*sin(omega*t) ----
    slosh_params: dict = field(default_factory=lambda: {
        'A': {'nom': 0.005, 'amp': 0.0, 'omega': 0.05},
        'B': {'nom': 0.020, 'amp': 0.0, 'omega': 0.05},
        'C': {'nom': 0.023, 'amp': 0.0, 'omega': 0.05},
        'K': {'nom': 0.008, 'amp': 0.0, 'omega': 0.05},
    })

    # ---- Slosh model (testing): rate-dependent stiffness/damping ----
    slosh_testing_params: dict = field(default_factory=lambda: {
        'omega_n':   0.08944,    # rad/s, sloshing-mode natural frequency
        'zeta':      0.1286,     # sloshing-mode damping ratio
        'omega_max': 0.418879,   # rad/s, max expected body rotation rate (SPICEsat: 24 deg/s)
    })

    # ---- Initial conditions ----
    q0:    np.ndarray = field(default_factory=lambda:
        (math.sqrt(2) / 2.0) * np.array([1.0, 0.0, 0.0, 1.0]))   # 90° about +X
    w0:    np.ndarray = field(default_factory=lambda:
        np.array([0.01, 0.01, 0.01]))                            # rad/s
    q_des: np.ndarray = field(default_factory=lambda:
        np.array([0.0, 0.0, 0.0, 1.0]))                          # target = identity
    Ts0:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    Tsd0:  np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ---- Time grid ----
    t_end:   float = 20 * 60.0    # seconds
    dt_eval: float = 0.05         # output sample step

    # ---- Toggles ----
    enable_slosh:        bool = False
    enable_magnetorquer: bool = False
    enable_wheels:       bool = True

    # ---- Solver settings ----
    method:   str   = 'RK45'      # 'RK45' | 'DOP853' | 'LSODA' | ...
    rtol:     float = 1e-8
    atol:     float = 1e-10
    max_step: float = 0.5         # cap solver step for time-varying terms


# ---------- Named presets ----------

def mc_example_71() -> SimConfig:
    """M&C Example 7.1 — large spacecraft, diagonal J, fast initial tumble."""
    cfg = SimConfig()
    cfg.Ixx, cfg.Iyy, cfg.Izz = 10000.0, 9000.0, 12000.0
    cfg.Ixy = cfg.Ixz = cfg.Iyz = 0.0
    cfg.Kp, cfg.Kd = 50.0, 500.0
    cfg.u_max = 100.0
    q = np.array([0.6853, 0.6953, 0.1531, 0.1531])
    cfg.q0 = q / np.linalg.norm(q)
    cfg.w0 = np.array([0.53, 0.53, 0.053])      # rad/s
    cfg.t_end = 300.0
    return cfg


def mc_example_72() -> SimConfig:
    """M&C Example 7.2 — smaller spacecraft w/ products of inertia, gentle IC.
    All SimConfig() defaults already match Example 7.2, so this is just an
    alias that makes intent explicit at the call site."""
    return SimConfig()


# =====================================================================
# Quaternion helpers (scalar-last convention)
# =====================================================================

def quat_multiply(q, p):
    """q ⊗ p  with scalar-last convention (M&C Eq. 2.82b)."""
    qv, q4 = q[0:3], q[3]
    pv, p4 = p[0:3], p[3]
    vec = q4 * pv + p4 * qv - np.cross(qv, pv)
    sca = q4 * p4 - np.dot(qv, pv)
    return np.hstack((vec, sca))


def quat_inverse(q):
    """Inverse (conjugate) of a unit quaternion, scalar last."""
    return np.hstack((-q[0:3], q[3]))


def quat_kinematics(q, omega):
    """
    Quaternion rate:  q_dot = (1/2) * Omega(omega) * q   (M&C Eq. 2.88).
        q_v_dot = (1/2) ( q4 * omega  -  omega x q_v )
        q_4_dot = -(1/2) omega · q_v
    """
    qv, q4 = q[0:3], q[3]
    qv_dot = 0.5 * (q4 * omega - np.cross(omega, qv))
    q4_dot = -0.5 * np.dot(omega, qv)
    return np.hstack((qv_dot, q4_dot))


def euler321_to_quat(roll_deg, pitch_deg, yaw_deg):
    """
    Convert a 3-2-1 (yaw-pitch-roll, Z-Y-X) Euler triple to a scalar-last
    quaternion (M&C App. B.96):  q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll).
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
    """
    if h_w_body is None:
        h_w_body = np.zeros(3)
    return np.linalg.solve(J, u_body - np.cross(omega, J @ omega + h_w_body))


def allocate_wheel_torque(u_cmd, W_pinv):
    """
    Pseudoinverse allocation of a body torque command to per-wheel torques:
        -W @ tau_w = u_cmd     =>     tau_w = -W^+ @ u_cmd
    """
    return -W_pinv @ u_cmd


# =====================================================================
# Magnetic field and B-dot momentum-dumping controller
# =====================================================================

def magnetic_field_body(t):
    """
    Toy Earth B-field in the body frame, ~30 µT magnitude, rotating with
    one 90-minute LEO orbital period.  Replace with IGRF + attitude
    rotation for a high-fidelity simulation.
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
    """
    B2 = float(np.dot(B_body, B_body))
    if B2 < 1e-14:                          # avoid divide-by-zero
        return np.zeros(3)
    m_cmd = K_dump * np.cross(h_w_body, B_body) / B2
    return np.clip(m_cmd, -m_max, m_max)


# =====================================================================
# Propellant slosh model
# =====================================================================

def slosh_coefficient(t, nom, amp, omega):
    """
    Time-varying scalar coefficient with a DC offset:
        f(t) = nom + amp * sin(omega * t)
    Keep nom > |amp| so K_s, C_s never go negative (which would be
    unphysical — negative stiffness or damping).
    """
    return nom + amp * np.sin(omega * t)


def slosh_dynamics_bourdelle(omega_body, omega_dot, Ts, Ts_dot, t, slosh_params):
    """
    Second-order propellant-slosh model driven by spacecraft body motion:
        Ts_ddot = -A_s(t)*omega - B_s(t)*omega_dot
                  - C_s(t)*Ts_dot - K_s(t)*Ts
    Returns Ts_ddot (3,).  `slosh_params` is a dict keyed by 'A','B','C','K',
    each holding {'nom':..., 'amp':..., 'omega':...}.
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
    """
    omega_n   = testing_params['omega_n']
    zeta      = testing_params['zeta']
    omega_max = testing_params['omega_max']

    K = omega_n ** 2
    ratio_sq = (omega_body / omega_max) ** 2

    return (-K * Ts
            - (2.0 * zeta * omega_n / (1.0 + ratio_sq)) * Ts_dot
            + K * (1.0 + ratio_sq) * omega_dot)


# =====================================================================
# Quaternion PD attitude controller
# =====================================================================

def pd_controller(q, omega, q_des, Kp, Kd, u_max=np.inf):
    """
    Quaternion-feedback PD law (M&C Sec. 7.4), with per-axis saturation:
        u = clip( -Kp * sign(dq4) * dq_vec  -  Kd * omega,  ±u_max )
    where dq = q_des^(-1) ⊗ q.  The sign(dq4) factor selects the shortest
    rotation (prevents 'unwinding').  Returns (u, dq).
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

def pack_state(q, w, h_w, Ts, Tsd):
    return np.concatenate([q, w, h_w, Ts, Tsd])


def unpack_state(y, n_w):
    q   = y[0:4]
    w   = y[4:7]
    h_w = y[7:7 + n_w]
    Ts  = y[7 + n_w : 10 + n_w]
    Tsd = y[10 + n_w : 13 + n_w]
    return q, w, h_w, Ts, Tsd


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
    q, w, h_w, Ts, Tsd = unpack_state(y, params['n_w'])
    W      = params['W']
    W_pinv = params['W_pinv']

    # 1) Magnetorquer (toggle)
    if params['enable_magnetorquer']:
        B_body  = magnetic_field_body(t)
        m_cmd   = bdot_momentum_dump(W @ h_w, B_body,
                                     params['K_dump'], params['m_max'])
        tau_mag = np.cross(m_cmd, B_body)
    else:
        tau_mag = np.zeros(3)

    # 2) PD attitude controller -> desired body torque
    u_cmd, _ = pd_controller(q, w, params['q_des'],
                             params['Kp'], params['Kd'], params['u_max'])

    # 3) Actuator branch: reaction wheels (toggle) or ideal torque actuator.
    #    With wheels OFF, the PD output drives the body directly and the wheel
    #    state is frozen (no momentum accumulation, no reaction torque).
    if params['enable_wheels']:
        tau_w  = allocate_wheel_torque(u_cmd, W_pinv)
        u_act  = -W @ tau_w                # wheel reaction on the body
        h_body = W @ h_w                   # wheel momentum in body frame
    else:
        tau_w  = np.zeros(params['n_w'])
        u_act  = u_cmd                     # ideal body-torque actuator
        h_body = np.zeros(3)

    # 4) Slosh torque on the body (zero if disabled)
    slosh_torque = Ts if params['enable_slosh'] else np.zeros(3)

    # 5) Euler's equation -> body angular acceleration
    u_body = u_act + tau_mag + slosh_torque
    w_dot  = rigid_body_dynamics(w, params['J'], u_body, h_body)

    # 6) Quaternion kinematics and wheel-momentum law
    q_dot = quat_kinematics(q, w)
    h_dot = tau_w                          # zero when wheels disabled

    # 7) Slosh second-order ODE (uses w_dot just computed)
    if params['enable_slosh']:
        Ts_dot = Tsd
        if params['slosh_model'] == 'testing':
            Tsd_dot = slosh_dynamics_testing(w, w_dot, Ts, Tsd,
                                             params['slosh_testing_params'])
        else:
            Tsd_dot = slosh_dynamics_bourdelle(w, w_dot, Ts, Tsd, t,
                                               params['slosh_params'])
    else:
        Ts_dot  = np.zeros(3)
        Tsd_dot = np.zeros(3)

    return np.concatenate([q_dot, w_dot, h_dot, Ts_dot, Tsd_dot])


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
    J = np.array([[ cfg.Ixx, -cfg.Ixy, -cfg.Ixz],
                  [-cfg.Ixy,  cfg.Iyy, -cfg.Iyz],
                  [-cfg.Ixz, -cfg.Iyz,  cfg.Izz]])
    assert np.allclose(J, J.T), "Inertia tensor must be symmetric"
    assert np.all(np.linalg.eigvalsh(J) > 0), "Inertia tensor must be positive-definite"

    # ---- Reaction-wheel configuration: 4-wheel NASA pyramid (M&C Sec. 7.2) ----
    # Spin axes tilted `wheel_tilt_deg` from +Z, equally spaced 90° in azimuth.
    beta = np.deg2rad(cfg.wheel_tilt_deg)
    cb, sb = np.cos(beta), np.sin(beta)
    W = np.array([[ sb,   0.0, -sb,   0.0],
                  [ 0.0,  sb,   0.0, -sb ],
                  [ cb,   cb,   cb,   cb ]])           # (3 x n_w)
    n_w = W.shape[1]
    W_pinv = np.linalg.pinv(W)

    # ---- Initial state vector ----
    y0 = pack_state(cfg.q0.copy(),
                    cfg.w0.copy(),
                    np.zeros(n_w),
                    cfg.Ts0.copy(),
                    cfg.Tsd0.copy())

    # ---- Output time grid ----
    t_eval = np.arange(0.0, cfg.t_end + cfg.dt_eval, cfg.dt_eval)

    # ---- Parameter bundle for the RHS closure ----
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
    Y = sol.y.T            # (N, 13 + n_w)
    N = len(t)

    # ---- Extract state trajectories ----
    q_hist  = Y[:, 0:4]
    w_hist  = Y[:, 4:7]
    hw_hist = Y[:, 7:7 + n_w]
    Ts_hist = Y[:, 7 + n_w : 10 + n_w]
    # Re-normalize quaternions (solve_ivp drift is small but nonzero).
    q_hist = q_hist / np.linalg.norm(q_hist, axis=1, keepdims=True)

    # ---- Reconstruct controller telemetry at each sample ----
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
        tau_w = (allocate_wheel_torque(u_cmd, W_pinv)
                 if cfg.enable_wheels else np.zeros(n_w))

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
                 hw_hist, tauw_hist, m_hist, tmag_hist, B_hist, Ts_hist):
    """Main 9-panel telemetry figure: states + controller outputs + slosh."""
    fig, ax = plt.subplots(9, 1, figsize=(10, 18), sharex=True)

    ax[0].plot(t, q_hist)
    ax[0].set_ylabel("Quaternion")
    ax[0].legend(["q1", "q2", "q3", "q4 (scalar)"], loc="right")
    ax[0].grid(True)

    ax[1].plot(t, dq_hist)
    ax[1].set_ylabel("Error quaternion")
    ax[1].legend(["dq1", "dq2", "dq3", "dq4 (scalar)"], loc="right")
    ax[1].grid(True)

    ax[2].plot(t, w_hist)
    ax[2].set_ylabel("Body rate (rad/s)")
    ax[2].legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"], loc="right")
    ax[2].grid(True)

    ax[3].plot(t, u_hist)
#    ax[3].set_ylabel("Cmd body torque (N·m)")
    ax[3].set_ylabel(r"$T_c$ (N·m)")
    ax[3].legend([r"$T_{c,x}$", r"$T_{c,y}$", r"$T_{c,z}$"], loc="right")
    ax[3].grid(True)

    n_w = hw_hist.shape[1]
    wheel_labels = [f"wheel {i+1}" for i in range(n_w)]

    ax[4].plot(t, hw_hist)
    ax[4].set_ylabel("Wheel momentum (N·m·s)")
    ax[4].legend(wheel_labels, loc="right")
    ax[4].grid(True)

    ax[5].plot(t, tauw_hist)
    ax[5].set_ylabel("Wheel torque (N·m)")
    ax[5].legend(wheel_labels, loc="right")
    ax[5].grid(True)

    ax[6].plot(t, m_hist)
    ax[6].set_ylabel("Mag. dipole (A·m²)")
    ax[6].legend([r"$m_x$", r"$m_y$", r"$m_z$"], loc="right")
    ax[6].grid(True)

    ax[7].plot(t, tmag_hist)
    ax[7].set_ylabel("Mag. torque (N·m)")
    ax[7].legend([r"$\tau_{mag,x}$", r"$\tau_{mag,y}$", r"$\tau_{mag,z}$"], loc="right")
    ax[7].grid(True)

    ax[8].plot(t, Ts_hist)
    ax[8].set_ylabel("Slosh torque $T_s$ (N·m)")
    ax[8].set_xlabel("Time (s)")
    ax[8].legend([r"$T_{s,x}$", r"$T_{s,y}$", r"$T_{s,z}$"], loc="right")
    ax[8].grid(True)

    plt.tight_layout()


def plot_euler_and_rates(t, q_hist, w_hist):
    """Second figure: 3-2-1 Euler angles (deg) and body rates (deg/s)."""
    N = len(t)
    eul_deg = np.zeros((N, 3))
    for k in range(N):
        roll, pitch, yaw = quat_to_euler321(q_hist[k])
        eul_deg[k] = np.rad2deg([roll, pitch, yaw])
    w_deg = np.rad2deg(w_hist)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(t, eul_deg)
    ax[0].set_ylabel("Euler angle (deg)")
    ax[0].legend(["roll  ($\\phi$)", "pitch ($\\theta$)", "yaw  ($\\psi$)"], loc="right")
    ax[0].grid(True)
    ax[0].set_title("Attitude angles and body rates (3-2-1 sequence)")

    ax[1].plot(t, w_deg)
    ax[1].set_ylabel("Body rate (deg/s)")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"], loc="right")
    ax[1].grid(True)

    plt.tight_layout()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    # ---- Pick a preset, then tweak any field you want before running ----
    # Available presets:
    #     SimConfig()         # baseline (= M&C Example 7.2)
    #     mc_example_71()
    #     mc_example_72()
    cfg = mc_example_72()

    # Examples of in-place tweaks:
    cfg.enable_slosh = True
    # cfg.enable_magnetorquer = True
    # cfg.enable_wheels = False
    # cfg.Kp, cfg.Kd = 15.0, 200.0
    # cfg.t_end = 600.0

    results = simulate(cfg)
    plot_results(*results)
    plot_euler_and_rates(results[0], results[1], results[3])
    plt.show()                                  # open both figures together
