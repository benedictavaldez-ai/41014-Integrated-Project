import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.perception.aruco import ArucoDetector
from src.perception.lidar import Lidar

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "world.xml"

WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.288
SENSOR_PERIOD = 0.1
PRINT_PERIOD = 0.5


def wheel_speeds(v, w):
    left = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    right = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    return left, right


def open_loop_command(t):
    if t < 6.5:
        return 0.2, 0.0
    if t < 8.23:
        return 0.0, 0.6
    if t < 22.0:
        return 0.2, 0.0
    return 0.0, 0.0


def run(duration, headless):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    lidar = Lidar(model)
    renderer = mujoco.Renderer(model, 480, 640)
    aruco = ArucoDetector()

    viewer = None if headless else mujoco.viewer.launch_passive(model, data)
    next_sensor = 0.0
    next_print = 0.0
    ranges = np.array([np.inf])
    markers = []
    wall_start = time.time()

    while data.time < duration and (viewer is None or viewer.is_running()):
        v, w = open_loop_command(data.time)
        data.ctrl[:] = wheel_speeds(v, w)
        mujoco.mj_step(model, data)

        if data.time >= next_sensor:
            next_sensor += SENSOR_PERIOD
            _, ranges = lidar.scan(data)
            renderer.update_scene(data, camera="pi_cam")
            markers = aruco.detect(renderer.render())

        if data.time >= next_print:
            next_print += PRINT_PERIOD
            enc_left, enc_right = data.sensor("enc_left").data[0], data.sensor("enc_right").data[0]
            marker_text = ", ".join(f"id {m['id']} r={m['range']:.2f} b={np.rad2deg(m['bearing']):.1f}deg" for m in markers) or "none"
            print(f"t={data.time:5.2f}  enc L={enc_left:7.2f} R={enc_right:7.2f} rad  lidar min={np.min(ranges):.2f} m  markers: {marker_text}")

        if viewer is not None:
            viewer.sync()
            lag = data.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(args.duration, args.headless)
