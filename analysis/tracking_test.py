import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.ground_truth import true_pose
from main import MODEL_PATH, wheel_speeds
from src.control.lqr import LQRTracker, tracking_error
from src.control.safety import front_clearance, limit_speed
from src.control.trajectory import Trajectory
from src.estimation.ekf import EKF, wrap
from src.perception.aruco import ArucoDetector
from src.perception.encoders import Encoders
from src.perception.lidar import Lidar
from src.perception.marker_pose import marker_measurement

START_POSE = np.array([-1.5, -1.5, 0.0])
DOCK_POSE = np.array([1.70, 1.20, 0.0])
CONTROL_PERIOD = 0.02
SENSOR_PERIOD = 0.1
SETTLE_TIME = 3.0


def run(init_error, seed):
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    aruco = ArucoDetector()
    lidar = Lidar(model, rng=rng)
    enc = Encoders(data)
    traj = Trajectory(CONTROL_PERIOD)
    tracker = LQRTracker(CONTROL_PERIOD)

    # initial estimate
    P0 = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(3) ** 2])
    x0 = START_POSE + (rng.multivariate_normal(np.zeros(3), P0) if init_error else 0)
    ekf = EKF(x0, P0)

    log = {k: [] for k in ["t", "ref", "est", "true", "P", "cmd", "err", "clear"]}
    clearance = np.inf
    next_control, next_sensor = 0.0, 0.0
    while data.time < traj.duration + SETTLE_TIME:
        if data.time >= next_sensor:
            next_sensor += SENSOR_PERIOD
            angles, ranges = lidar.scan(data)
            clearance = front_clearance(angles, ranges)
            renderer.update_scene(data, camera="pi_cam")
            for det in aruco.detect(renderer.render()):
                z, R = marker_measurement(det)
                ekf.update_marker(z, R, det["id"])

        if data.time >= next_control:
            next_control += CONTROL_PERIOD
            ekf.predict(enc.delta())
            ref = traj.sample(data.time)
            if data.time < traj.duration:
                (v, w), e = tracker.command(ekf.x, ref)
                v = limit_speed(v, clearance)
            else:
                v, w, e = 0.0, 0.0, tracking_error(ekf.x, ref)
            data.ctrl[:] = wheel_speeds(v, w)

            log["t"].append(data.time)
            log["ref"].append(ref)
            log["est"].append(ekf.x.copy())
            log["true"].append(true_pose(data))
            log["P"].append(ekf.P.copy())
            log["cmd"].append((v, w))
            log["err"].append(e)
            log["clear"].append(clearance)

        mujoco.mj_step(model, data)

    out = {k: np.array(v) for k, v in log.items()}
    out["traj"] = traj
    return out


def report(log):
    true_err = np.array([tracking_error(x, r) for x, r in zip(log["true"], log["ref"])])
    est_err = log["est"] - log["true"]
    est_err[:, 2] = wrap(est_err[:, 2])
    moving = log["t"] < log["traj"].duration
    dock = log["true"][-1] - DOCK_POSE
    dock[2] = wrap(dock[2])
    wheels = np.array([wheel_speeds(v, w) for v, w in log["cmd"]])
    print(f"path {log['traj'].length:.2f} m, reference duration {log['traj'].duration:.1f} s")
    print(f"tracking RMSE (true vs ref): along {np.sqrt(np.mean(true_err[moving, 0] ** 2)):.3f} m  "
          f"lateral {np.sqrt(np.mean(true_err[moving, 1] ** 2)):.3f} m  heading {np.rad2deg(np.sqrt(np.mean(true_err[moving, 2] ** 2))):.2f} deg")
    print(f"max lateral error {np.abs(true_err[moving, 1]).max():.3f} m")
    print(f"estimation RMSE: pos {np.sqrt(np.mean(np.sum(est_err[:, :2] ** 2, 1))):.3f} m  heading {np.rad2deg(np.sqrt(np.mean(est_err[:, 2] ** 2))):.2f} deg")
    print(f"final docking error (true): pos {np.hypot(*dock[:2]):.3f} m  (x {dock[0]:+.3f}, y {dock[1]:+.3f})  heading {np.rad2deg(dock[2]):+.2f} deg")
    print(f"min front LiDAR clearance {np.min(log['clear']):.3f} m")
    print(f"peak wheel speed {np.abs(wheels).max():.2f} rad/s (limit 7.88), RMS w {np.sqrt(np.mean(log['cmd'][:, 1] ** 2)):.3f} rad/s")
    return true_err, est_err


def plot(log, true_err, est_err, out):
    t, traj = log["t"], log["traj"]
    sig = np.sqrt(np.array([np.diag(P) for P in log["P"]]))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # path
    a = ax[0, 0]
    for x0, y0, w, h in [(-0.8, -0.9, 0.4, 0.8), (0.5, -1.2, 0.6, 0.4), (1.9, 1.0, 0.1, 0.4)]:
        a.add_patch(mpatches.Rectangle((x0, y0), w, h, color="0.6"))
    for cx, cy, r in [(0.2, 0.6, 0.2), (-1.0, 1.0, 0.15)]:
        a.add_patch(mpatches.Circle((cx, cy), r, color="0.6"))
    a.plot(*traj.xy_path.T, "k--", lw=1, label="reference")
    a.plot(*log["true"][:, :2].T, "g", label="true")
    a.plot(*log["est"][:, :2].T, "b:", label="EKF estimate")
    a.plot(*DOCK_POSE[:2], "r*", ms=10, label="dock point")
    a.set_aspect("equal"); a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.legend(fontsize=8, loc="upper left"); a.set_title("path")

    # tracking errors
    names = ["along-track error (m)", "lateral error (m)", "heading error (deg)"]
    scale = [1, 1, np.rad2deg(1)]
    for i, a in enumerate([ax[0, 1], ax[0, 2], ax[1, 0]]):
        a.plot(t, true_err[:, i] * scale[i], "g", label="true vs reference")
        a.plot(t, log["err"][:, i] * scale[i], "b:", label="estimate vs reference (controller sees)")
        a.axvline(traj.duration, color="k", lw=0.8, ls="--")
        a.set_title(names[i]); a.set_xlabel("t (s)")
    ax[0, 1].legend(fontsize=8)

    # commands
    a = ax[1, 1]
    a.plot(t, log["ref"][:, 3], "k--", lw=1, label="v ref")
    a.plot(t, log["cmd"][:, 0], "b", label="v cmd")
    a.plot(t, log["ref"][:, 4], "--", color="0.5", lw=1, label="w ref")
    a.plot(t, log["cmd"][:, 1], "r", label="w cmd")
    a.set_title("commands (m/s, rad/s)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)

    # estimation error
    a = ax[1, 2]
    pos_err = np.hypot(est_err[:, 0], est_err[:, 1])
    a.plot(t, pos_err, "b", label="EKF position error")
    a.plot(t, 3 * np.hypot(sig[:, 0], sig[:, 1]), "b", alpha=0.3, label="3σ")
    a.set_title("estimation error (m)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init-error", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    log = run(args.init_error, args.seed)
    name = f"tracking_init_seed{args.seed}.png" if args.init_error else "tracking.png"
    plot(log, *report(log), ROOT / "docs" / name)
