import numpy as np

from src.estimation.ekf import WHEEL_RADIUS, WHEEL_SEPARATION, wrap

# LQR weights
E_MAX = np.array([0.10, 0.05, np.deg2rad(10)])
U_MAX = np.array([0.10, 0.50])
Q = np.diag(1 / E_MAX ** 2)
R = np.diag(1 / U_MAX ** 2)

# limits
WHEEL_LIMIT = 7.88
V_LIN_MIN = 0.05


# tracking error in the robot frame
def tracking_error(x, ref):
    dx, dy = ref[0] - x[0], ref[1] - x[1]
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(ref[2] - x[2])])


# discrete error model
def error_model(v_r, w_r, dt):
    v = max(v_r, V_LIN_MIN)
    A = np.array([[0, w_r, 0], [-w_r, 0, v], [0, 0, 0]])
    B = np.array([[1, 0], [0, 0], [0, 1]])
    return np.eye(3) + A * dt, B * dt


# discrete Riccati equation
def dlqr(A, B, Q, R, iters=50, tol=1e-10):
    n = A.shape[0]
    Ak, Gk, Hk = A.copy(), B @ np.linalg.solve(R, B.T), Q.copy()
    for _ in range(iters):
        W = np.linalg.inv(np.eye(n) + Gk @ Hk)
        A_next = Ak @ W @ Ak
        G_next = Gk + Ak @ W @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ W @ Ak
        done = np.max(np.abs(H_next - Hk)) < tol * max(1.0, np.max(np.abs(Hk)))
        Ak, Gk, Hk = A_next, G_next, H_next
        if done:
            break
    P = Hk
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# wheel saturation
def saturate(v, w):
    left = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    right = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    k = max(abs(left), abs(right)) / WHEEL_LIMIT
    return (v / k, w / k) if k > 1 else (v, w)


class LQRTracker:
    def __init__(self, dt, Q=Q, R=R):
        self.dt, self.Q, self.R = dt, Q, R
        self.K = None

    # command from estimate and reference
    def command(self, x_est, ref):
        e = tracking_error(x_est, ref)
        A, B = error_model(ref[3], ref[4], self.dt)
        self.K, _ = dlqr(A, B, self.Q, self.R)
        du = -self.K @ e
        v = ref[3] * np.cos(e[2]) - du[0]
        w = ref[4] - du[1]
        return saturate(v, w), e
