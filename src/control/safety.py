import numpy as np

# front sector and distances
SECTOR = np.deg2rad(30)
STOP_DIST = 0.18
SLOW_DIST = 0.35
V_MAX = 0.26

# distances while docking
DOCK_STOP_DIST = 0.12
DOCK_SLOW_DIST = 0.25


# front clearance
def front_clearance(angles, ranges):
    front = np.abs(angles) <= SECTOR
    return float(np.min(ranges[front])) if np.any(front) else np.inf


# forward speed cap
def limit_speed(v, clearance, stop=STOP_DIST, slow=SLOW_DIST):
    cap = V_MAX * np.clip((clearance - stop) / (slow - stop), 0.0, 1.0)
    return min(v, cap) if v > 0 else v
