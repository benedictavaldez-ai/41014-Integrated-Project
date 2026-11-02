import numpy as np

# front sector and distances
SECTOR = np.deg2rad(30)
STOP_DIST = 0.18
SLOW_DIST = 0.35
V_MAX = 0.26


# front clearance
def front_clearance(angles, ranges):
    front = np.abs(angles) <= SECTOR
    return float(np.min(ranges[front])) if np.any(front) else np.inf


# forward speed cap
def limit_speed(v, clearance):
    cap = V_MAX * np.clip((clearance - STOP_DIST) / (SLOW_DIST - STOP_DIST), 0.0, 1.0)
    return min(v, cap) if v > 0 else v
