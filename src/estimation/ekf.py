import numpy as np

from src.perception.marker_pose import CAM_POS, MARKER_MAP

WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.288

# odometry noise
WHEEL_NOISE_K = 0.02
WHEEL_NOISE_FLOOR = 1e-4

# innovation gates
GATE_RANGE_BEARING = 9.21
GATE_HEADING = 6.63


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.last_nis = None

    # predict step
    def predict(self, dphi):
        r, b = WHEEL_RADIUS, WHEEL_SEPARATION
        ds = r * (dphi[0] + dphi[1]) / 2
        dth = r * (dphi[1] - dphi[0]) / b
        th_mid = self.x[2] + dth / 2
        c, s = np.cos(th_mid), np.sin(th_mid)

        self.x = self.x + np.array([ds * c, ds * s, dth])
        self.x[2] = wrap(self.x[2])

        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1]])
        J = np.array([[c, -ds * s / 2], [s, ds * c / 2], [0, 1]]) @ np.array([[r / 2, r / 2], [-r / b, r / b]])
        sig = WHEEL_NOISE_K * np.abs(dphi) + WHEEL_NOISE_FLOOR
        Q = J @ np.diag(sig ** 2) @ J.T
        self.P = F @ self.P @ F.T + Q

    # range and bearing model
    def range_bearing_model(self, marker_id):
        mx, my, mz, _ = MARKER_MAP[marker_id]
        x, y, th = self.x
        d, h = CAM_POS[0], CAM_POS[2]
        dx = mx - (x + d * np.cos(th))
        dy = my - (y + d * np.sin(th))
        dz = mz - h
        q = dx ** 2 + dy ** 2
        rng = np.sqrt(q + dz ** 2)
        brg = wrap(np.arctan2(dy, dx) - th)
        ddx, ddy = d * np.sin(th), -d * np.cos(th)
        H = np.array([
            [-dx / rng, -dy / rng, (dx * ddx + dy * ddy) / rng],
            [dy / q, -dx / q, (-dy * ddx + dx * ddy) / q - 1],
        ])
        return np.array([rng, brg]), H

    # kalman update with gate
    def _update(self, y, H, R, gate):
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > gate:
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True, nis

    # marker update
    def update_marker(self, z, R, marker_id):
        z_hat, H = self.range_bearing_model(marker_id)
        y = z[:2] - z_hat
        y[1] = wrap(y[1])
        ok_rb, nis_rb = self._update(y, H, R[:2, :2], GATE_RANGE_BEARING)

        y_h = np.array([wrap(z[2] - self.x[2])])
        ok_h, nis_h = self._update(y_h, np.array([[0.0, 0.0, 1.0]]), R[2:, 2:], GATE_HEADING)
        self.last_nis = (nis_rb, nis_h)
        return ok_rb, ok_h
