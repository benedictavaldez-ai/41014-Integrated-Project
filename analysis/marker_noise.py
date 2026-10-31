import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import MODEL_PATH
from src.estimation.ekf import wrap
from src.perception.aruco import ArucoDetector
from src.perception.marker_pose import CAM_POS, MARKER_MAP, robot_pose_from_marker

N_SAMPLES = 400
MARKER_ID = 0


# place robot
def place_robot(model, data, x, y, th):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0:3] = [x, y, 0.0]
    data.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    mujoco.mj_forward(model, data)


# true range and bearing
def true_range_bearing(x, y, th):
    mx, my, mz, _ = MARKER_MAP[MARKER_ID]
    cx, cy = x + CAM_POS[0] * np.cos(th), y + CAM_POS[0] * np.sin(th)
    r = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2 + (mz - CAM_POS[2]) ** 2)
    b = wrap(np.arctan2(my - cy, mx - cx) - th)
    return r, b


def collect(seed=1):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    aruco = ArucoDetector()
    rng = np.random.default_rng(seed)
    mx, my = MARKER_MAP[MARKER_ID][:2]

    rows = []
    while len(rows) < N_SAMPLES:
        x, y = rng.uniform(-1.6, 1.5), rng.uniform(-1.8, 1.8)
        th = np.arctan2(my - y, mx - x) + rng.uniform(-0.4, 0.4)
        place_robot(model, data, x, y, th)
        renderer.update_scene(data, camera="pi_cam")
        dets = [d for d in aruco.detect(renderer.render()) if d["id"] == MARKER_ID]
        if not dets:
            continue
        r, b = true_range_bearing(x, y, th)
        heading = robot_pose_from_marker(dets[0])[2]
        rows.append((r, dets[0]["range"] - r, wrap(dets[0]["bearing"] - b), wrap(heading - th)))
    return np.array(rows)


def robust_std(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def main():
    a = collect()
    r, e_r, e_b, e_h = a.T

    # fits
    bias_k = np.sum(e_r * r ** 2) / np.sum(r ** 4)
    resid = e_r - bias_k * r ** 2
    bins = np.arange(0.5, 4.0, 0.5)
    centres, r_std, h_std = [], [], []
    for lo in bins:
        s = (r >= lo) & (r < lo + 0.5)
        if s.sum() >= 10:
            centres.append(lo + 0.25)
            r_std.append(resid[s].std())
            h_std.append(robust_std(e_h[s]))
    centres, r_std, h_std = map(np.array, (centres, r_std, h_std))
    r_std_k = np.sum(r_std * centres ** 2) / np.sum(centres ** 4)
    h_std_k = np.sum(h_std * centres) / np.sum(centres ** 2)

    print(f"{len(a)} detections, range {r.min():.2f} to {r.max():.2f} m")
    print(f"range bias      k = {bias_k:.4f}  (bias = k r^2)")
    print(f"range std       k = {r_std_k:.4f}  (std = k r^2)")
    print(f"bearing error   mean {np.rad2deg(e_b.mean()):+.3f} deg  std {np.rad2deg(e_b.std()):.3f} deg")
    print(f"heading std     k = {np.rad2deg(h_std_k):.2f} deg per m  (robust, std = k r)")
    print(f"heading outliers > 10 deg: {np.mean(np.abs(e_h) > np.deg2rad(10)):.0%}")
    print(" bin    range std   heading std")
    for c, rs, hs in zip(centres, r_std, h_std):
        print(f" {c:4.2f}   {rs:.4f} m    {np.rad2deg(hs):.2f} deg")

    # plots
    rr = np.linspace(0.5, 3.6, 50)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(r, e_r, ".", ms=3, label="samples")
    ax[0].plot(rr, bias_k * rr ** 2, "r", label=f"bias {bias_k:.4f} r²")
    ax[0].set(xlabel="true range (m)", ylabel="range error (m)", title="range")
    ax[0].legend(fontsize=8)
    ax[1].plot(r, np.rad2deg(e_b), ".", ms=3)
    ax[1].set(xlabel="true range (m)", ylabel="bearing error (deg)", title="bearing")
    ax[2].plot(r, np.rad2deg(e_h), ".", ms=3)
    ax[2].plot(rr, 3 * np.rad2deg(h_std_k) * rr, "r", label="±3σ fit")
    ax[2].plot(rr, -3 * np.rad2deg(h_std_k) * rr, "r")
    ax[2].set(xlabel="true range (m)", ylabel="heading error (deg)", title="robot heading from marker", ylim=(-30, 30))
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "marker_noise.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
