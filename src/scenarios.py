import mujoco
import numpy as np

from src.perception.encoders import Encoders

SCENARIOS = ("nominal", "odom_bias", "push", "occlusion", "all")

# disturbance settings
ODOM_BIAS = 0.01
PUSH_TIME = 15.0
PUSH_DURATION = 0.3
PUSH_FORCE = np.array([18.0, 0.0, 0.0])
OCCLUSION = (24.0, 32.0)

# random variation
BIAS_RANGE = (0.005, 0.015)
PUSH_TIME_RANGE = (12.0, 18.0)
OCCLUSION_START_RANGE = (20.0, 26.0)


# biased encoders
class BiasedEncoders(Encoders):
    def __init__(self, data, bias=ODOM_BIAS, wheel=0):
        self.bias, self.wheel = bias, wheel
        super().__init__(data)

    def read(self):
        a = super().read()
        a[self.wheel] *= 1 + self.bias
        return a


class Scenario:
    def __init__(self, name, model, rng=None):
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name}")
        self.name = name
        self.odom_bias = name in ("odom_bias", "all")
        self.push = name in ("push", "all")
        self.occlusion = name in ("occlusion", "all")
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

        # fixed or random settings
        self.bias, self.bias_wheel = ODOM_BIAS, 0
        self.push_time, self.push_force = PUSH_TIME, PUSH_FORCE.copy()
        self.occlusion_window = OCCLUSION
        if rng is not None:
            self.bias, self.bias_wheel = rng.uniform(*BIAS_RANGE), int(rng.integers(2))
            self.push_time = rng.uniform(*PUSH_TIME_RANGE)
            self.push_force = PUSH_FORCE * rng.choice([-1.0, 1.0])
            start = rng.uniform(*OCCLUSION_START_RANGE)
            self.occlusion_window = (start, start + OCCLUSION[1] - OCCLUSION[0])

    def encoders(self, data):
        return BiasedEncoders(data, self.bias, self.bias_wheel) if self.odom_bias else Encoders(data)

    # push
    def apply(self, data, t):
        on = self.push and self.push_time <= t < self.push_time + PUSH_DURATION
        data.xfrc_applied[self.body, :3] = self.push_force if on else 0.0

    # camera covered
    def camera_blind(self, t):
        return self.occlusion and self.occlusion_window[0] <= t < self.occlusion_window[1]

    # settings used
    def describe(self):
        return {
            "bias": self.bias if self.odom_bias else 0.0,
            "bias_wheel": ("left" if self.bias_wheel == 0 else "right") if self.odom_bias else "",
            "push_time": self.push_time if self.push else np.nan,
            "push_dir": float(np.sign(self.push_force[0])) if self.push else 0.0,
            "blind_start": self.occlusion_window[0] if self.occlusion else np.nan,
        }
