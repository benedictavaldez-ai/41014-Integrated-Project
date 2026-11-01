import numpy as np

# waypoints
WAYPOINTS = np.array([
    [-1.50, -1.50],
    [0.05, -1.50],
    [0.05, -0.20],
    [0.90, 0.45],
    [1.05, 1.20],
    [1.70, 1.20],
])
FILLET_RADIUS = 0.30

# speed profile
V_CRUISE = 0.15
V_DOCK = 0.08
DOCK_SLOW_DIST = 0.50
ACCEL = 0.10


# straight lines joined by circular arcs
def build_path(waypoints=WAYPOINTS, radius=FILLET_RADIUS, ds=0.005):
    pts = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        p0, p1, p2 = waypoints[i - 1], waypoints[i], waypoints[i + 1]
        d1 = (p1 - p0) / np.linalg.norm(p1 - p0)
        d2 = (p2 - p1) / np.linalg.norm(p2 - p1)
        turn = np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)
        t = radius * np.tan(abs(turn) / 2)
        a, b = p1 - d1 * t, p1 + d2 * t
        pts += list(_line(pts[-1], a, ds))
        n = np.array([-d1[1], d1[0]]) * np.sign(turn)
        c = a + n * radius
        a0 = np.arctan2(a[1] - c[1], a[0] - c[0])
        ang = np.linspace(a0, a0 + turn, max(int(abs(turn) * radius / ds), 2))
        pts += list(np.column_stack([c[0] + radius * np.cos(ang), c[1] + radius * np.sin(ang)])[1:])
    pts += list(_line(pts[-1], waypoints[-1], ds))
    return np.array(pts)


def _line(a, b, ds):
    n = max(int(np.linalg.norm(b - a) / ds), 1)
    return a + np.outer(np.linspace(0, 1, n + 1)[1:], b - a)


# reference trajectory sampled in time
class Trajectory:
    def __init__(self, dt=0.02):
        xy = build_path()
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
        self.length = s[-1]

        # speed along the path
        v_lim = np.where(s > self.length - DOCK_SLOW_DIST, V_DOCK, V_CRUISE)
        v_lim = np.minimum(v_lim, np.sqrt(2 * ACCEL * s + 1e-4))
        v_lim = np.minimum(v_lim, np.sqrt(2 * ACCEL * (self.length - s)))
        for i in range(len(s) - 2, -1, -1):
            v_lim[i] = min(v_lim[i], np.sqrt(v_lim[i + 1] ** 2 + 2 * ACCEL * seg[i]))

        # time along the path
        v_mid = np.maximum((v_lim[:-1] + v_lim[1:]) / 2, 1e-3)
        t = np.concatenate([[0], np.cumsum(seg / v_mid)])
        self.duration = t[-1]

        # resample at the control rate
        self.t = np.arange(0, self.duration, dt)
        self.x = np.interp(self.t, t, xy[:, 0])
        self.y = np.interp(self.t, t, xy[:, 1])
        self.th = np.interp(self.t, t, heading)
        self.v = np.interp(self.t, t, v_lim)
        self.w = np.gradient(self.th, dt)
        self.xy_path = xy

    # reference at time t
    def sample(self, t):
        if t >= self.duration:
            return np.array([self.x[-1], self.y[-1], self.th[-1], 0.0, 0.0])
        i = int(t / (self.t[1] - self.t[0]))
        return np.array([self.x[i], self.y[i], self.th[i], self.v[i], self.w[i]])
