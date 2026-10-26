import numpy as np
import mujoco


class Lidar:
    def __init__(self, model, site="lidar", body="base", n_beams=360, max_range=3.5, min_range=0.12, noise_std=0.01, rng=None):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        self.max_range = max_range
        self.min_range = min_range
        self.noise_std = noise_std
        self.rng = rng if rng is not None else np.random.default_rng()
        self._geomid = np.zeros(1, dtype=np.int32)

    def scan(self, data):
        origin = data.site_xpos[self.site_id].copy()
        rot = data.site_xmat[self.site_id].reshape(3, 3)
        ranges = np.full(self.angles.shape, np.inf)
        for i, a in enumerate(self.angles):
            direction = rot @ np.array([np.cos(a), np.sin(a), 0.0])
            d = mujoco.mj_ray(self.model, data, origin, direction, None, 1, self.body_id, self._geomid)
            if self.min_range <= d <= self.max_range:
                ranges[i] = d + self.rng.normal(0.0, self.noise_std)
        return self.angles.copy(), ranges
