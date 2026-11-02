import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.ground_truth import true_pose
from main import MODEL_PATH, open_loop_command, wheel_speeds
from src.estimation.ekf import EKF, wrap
from src.perception.aruco import ArucoDetector
from src.perception.encoders import Encoders
from src.perception.marker_pose import marker_measurement

START_POSE = np.array([-1.5, -1.5, 0.0])
PREDICT_PERIOD = 0.02
CAMERA_PERIOD = 0.1


def run(duration, init_error, seed):
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    aruco = ArucoDetector()
    enc = Encoders(data)

    # initial estimate
    P0 = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(3) ** 2])
    x0 = START_POSE + (rng.multivariate_normal(np.zeros(3), P0) if init_error else 0)
    ekf = EKF(x0, P0)
    odom = EKF(x0, P0)

    log = {k: [] for k in ["t", "x", "P", "odom", "true", "meas"]}
    next_predict, next_camera = PREDICT_PERIOD, CAMERA_PERIOD
    while data.time < duration:
        data.ctrl[:] = wheel_speeds(*open_loop_command(data.time))
        mujoco.mj_step(model, data)

        if data.time >= next_predict:
            next_predict += PREDICT_PERIOD
            dphi = enc.delta()
            ekf.predict(dphi)
            odom.predict(dphi)

        if data.time >= next_camera:
            next_camera += CAMERA_PERIOD
            renderer.update_scene(data, camera="pi_cam")
            for det in aruco.detect(renderer.render()):
                z, R = marker_measurement(det)
                ok_rb, ok_h = ekf.update_marker(z, R, det["id"])
                log["meas"].append((data.time, ok_rb, ok_h))
            log["t"].append(data.time)
            log["x"].append(ekf.x.copy())
            log["P"].append(ekf.P.copy())
            log["odom"].append(odom.x.copy())
            log["true"].append(true_pose(data))

    return {k: np.array(v) for k, v in log.items()}


def report(log):
    err = log["x"] - log["true"]
    err[:, 2] = wrap(err[:, 2])
    oerr = log["odom"] - log["true"]
    oerr[:, 2] = wrap(oerr[:, 2])
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(err, log["P"])])
    m = log["meas"]
    print(f"marker updates: {len(m)}  range/bearing accepted {m[:, 1].mean():.0%}  heading accepted {m[:, 2].mean():.0%}" if len(m) else "no marker seen")
    print(f"final error   EKF: pos {np.hypot(*err[-1, :2]):.3f} m  heading {np.rad2deg(err[-1, 2]):+.2f} deg")
    print(f"final error  odom: pos {np.hypot(*oerr[-1, :2]):.3f} m  heading {np.rad2deg(oerr[-1, 2]):+.2f} deg")
    print(f"RMSE          EKF: pos {np.sqrt(np.mean(np.sum(err[:, :2] ** 2, 1))):.3f} m  heading {np.rad2deg(np.sqrt(np.mean(err[:, 2] ** 2))):.2f} deg")
    print(f"mean NEES {nees.mean():.2f} (consistent filter: 3.0), inside 95% bound {np.mean(nees < 7.81):.0%}")
    return err, oerr, nees


def plot(log, err, oerr, nees, out):
    t = log["t"]
    sig = np.sqrt(np.array([np.diag(P) for P in log["P"]]))
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))

    # path
    a = ax[0, 0]
    a.plot(*log["true"][:, :2].T, "k", label="true")
    a.plot(*log["x"][:, :2].T, "b--", label="EKF")
    a.plot(*log["odom"][:, :2].T, "r:", label="odometry only")
    a.plot(1.899, 1.2, "gs", label="marker")
    a.set_aspect("equal"); a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.legend(fontsize=8); a.set_title("path")

    # errors
    names = ["x error (m)", "y error (m)", "heading error (deg)"]
    scale = [1, 1, np.rad2deg(1)]
    for i, a in enumerate([ax[0, 1], ax[0, 2], ax[1, 0]]):
        a.plot(t, err[:, i] * scale[i], "b", label="EKF error")
        a.fill_between(t, -3 * sig[:, i] * scale[i], 3 * sig[:, i] * scale[i], color="b", alpha=0.15, label="±3σ")
        a.plot(t, oerr[:, i] * scale[i], "r:", label="odometry error")
        a.set_title(names[i]); a.set_xlabel("t (s)")
    ax[0, 1].legend(fontsize=8)

    # nees
    a = ax[1, 1]
    a.semilogy(t, nees, "b")
    a.axhline(7.81, color="k", ls="--", label="95% bound (3 dof)")
    a.axhline(3.0, color="g", ls=":", label="expected value")
    a.set_title("NEES"); a.set_xlabel("t (s)"); a.legend(fontsize=8)

    # marker updates
    a = ax[1, 2]
    m = log["meas"]
    if len(m):
        a.plot(m[:, 0], m[:, 1], "g.", label="range/bearing accepted")
        a.plot(m[:, 0], m[:, 2] * 0.9, "m.", label="heading accepted")
    a.set_ylim(-0.2, 1.2); a.set_title("marker updates (1 = accepted)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=23.0)
    p.add_argument("--init-error", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    log = run(args.duration, args.init_error, args.seed)
    name = f"ekf_open_loop_init_seed{args.seed}.png" if args.init_error else "ekf_open_loop.png"
    plot(log, *report(log), ROOT / "docs" / name)
