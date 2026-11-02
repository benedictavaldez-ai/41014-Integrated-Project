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


# biased encoders
class BiasedEncoders(Encoders):
    def read(self):
        a = super().read()
        a[0] *= 1 + ODOM_BIAS
        return a


class Scenario:
    def __init__(self, name, model):
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name}")
        self.name = name
        self.odom_bias = name in ("odom_bias", "all")
        self.push = name in ("push", "all")
        self.occlusion = name in ("occlusion", "all")
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    def encoders(self, data):
        return BiasedEncoders(data) if self.odom_bias else Encoders(data)

    # push
    def apply(self, data, t):
        on = self.push and PUSH_TIME <= t < PUSH_TIME + PUSH_DURATION
        data.xfrc_applied[self.body, :3] = PUSH_FORCE if on else 0.0

    # camera covered
    def camera_blind(self, t):
        return self.occlusion and OCCLUSION[0] <= t < OCCLUSION[1]
