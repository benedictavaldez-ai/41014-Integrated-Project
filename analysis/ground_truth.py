import numpy as np

# evaluation only


def true_pose(data):
    x, y = data.qpos[0], data.qpos[1]
    w, qx, qy, qz = data.qpos[3:7]
    theta = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return np.array([x, y, theta])
