import cv2
import numpy as np


class ArucoDetector:
    def __init__(self, marker_length=0.15, width=640, height=480, fovy_deg=48.8, dictionary=cv2.aruco.DICT_4X4_50):
        self.marker_length = marker_length
        f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros(5)
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary), cv2.aruco.DetectorParameters())
        h = marker_length / 2
        self.object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def detect(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is None:
            return results
        for c, marker_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(self.object_points, c.reshape(4, 2), self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            x_cam, y_cam, z_cam = tvec.flatten()
            results.append({
                "id": int(marker_id),
                "range": float(np.linalg.norm(tvec)),
                "bearing": float(-np.arctan2(x_cam, z_cam)),
                "tvec": tvec.flatten(),
                "rvec": rvec.flatten(),
            })
        return results
