import cv2
import numpy as np

# marker map
MARKER_MAP = {0: (1.899, 1.2, 0.13, np.pi)}

# camera pose on the robot
CAM_POS = np.array([0.086, 0.0, 0.107])
CAM_ROT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)

# measurement noise
RANGE_BIAS_K = 0.009
RANGE_SIGMA_K = 0.005
BEARING_SIGMA = np.deg2rad(0.1)
HEADING_SIGMA_K = np.deg2rad(0.9)
HEADING_SIGMA_MIN = np.deg2rad(1.0)

# noise inflation
INFLATION = 3.0


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


T_BASE_CAM = make_T(CAM_ROT, CAM_POS)


# marker frame in world
def marker_in_world(marker_id):
    x, y, z, yaw = MARKER_MAP[marker_id]
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, n)
    return make_T(np.column_stack([right, up, n]), [x, y, z])


# robot pose from one detection
def robot_pose_from_marker(det):
    R_cm, _ = cv2.Rodrigues(np.asarray(det["rvec"], dtype=float))
    T_cam_marker = make_T(R_cm, det["tvec"])
    T_world_base = marker_in_world(det["id"]) @ np.linalg.inv(T_cam_marker) @ np.linalg.inv(T_BASE_CAM)
    theta = np.arctan2(T_world_base[1, 0], T_world_base[0, 0])
    return np.array([T_world_base[0, 3], T_world_base[1, 3], theta])


# marker measurement and noise
def marker_measurement(det):
    r = det["range"] - RANGE_BIAS_K * det["range"] ** 2
    heading = robot_pose_from_marker(det)[2]
    z = np.array([r, det["bearing"], heading])
    sig = np.array([RANGE_SIGMA_K * r ** 2, BEARING_SIGMA, max(HEADING_SIGMA_MIN, HEADING_SIGMA_K * r)])
    return z, np.diag((INFLATION * sig) ** 2)
