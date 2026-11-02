import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from analysis.ground_truth import true_pose
from src.control.lqr import tracking_error
from src.estimation.ekf import wrap
from src.pipeline import DOCK_POSE, run
from src.scenarios import OCCLUSION, PUSH_TIME, SCENARIOS

ROOT = Path(__file__).resolve().parent


# evaluation only
class TruthRecorder:
    def __init__(self):
        self.poses = []

    def __call__(self, data):
        self.poses.append(true_pose(data))


def summarise(log, truth):
    moving = log["ref_clock"] < log["traj"].duration
    true_err = np.array([tracking_error(x, r) for x, r in zip(truth, log["ref"])])
    est_err = log["est"] - truth
    est_err[:, 2] = wrap(est_err[:, 2])
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(est_err, log["P"])])
    dock = truth[-1] - DOCK_POSE
    dock[2] = wrap(dock[2])
    seen = np.array(log["marker_ok"]) & (np.array([np.hypot(*(DOCK_POSE[:2] - x[:2])) for x in log["est"]]) > 0.5)
    first_fix = log["t"][np.argmax(seen)] if np.any(seen) else np.inf
    dock_start = log["t"][np.argmax(log["state"] == "DOCK")] if np.any(log["state"] == "DOCK") else np.nan
    res = {
        "scenario": log["scenario"],
        "final_state": log["final_state"],
        "docked": bool(log["docked"]),
        "time": log["t"][-1],
        "dock_start": dock_start,
        "acquire": np.sum(log["state"] == "ACQUIRE") * (log["t"][1] - log["t"][0]),
        "paused": np.sum(log["paused"]) * (log["t"][1] - log["t"][0]),
        "lateral_rmse": np.sqrt(np.mean(true_err[moving, 1] ** 2)),
        "lateral_max": np.abs(true_err[moving, 1]).max(),
        "heading_rmse": np.rad2deg(np.sqrt(np.mean(true_err[moving, 2] ** 2))),
        "est_rmse": np.sqrt(np.mean(np.sum(est_err[:, :2] ** 2, 1))),
        "est_max": np.hypot(est_err[:, 0], est_err[:, 1]).max(),
        "nees": nees.mean(),
        "dock_pos": np.hypot(*dock[:2]),
        "dock_heading": np.rad2deg(dock[2]),
        "min_clear": np.min(log["clear"]),
        "false_ids": log["false_ids"],
        "relocalised": log["relocalised"],
        "bad_pnp": log["bad_pnp"],
        "nees_marker": nees[log["t"] >= first_fix].mean() if np.isfinite(first_fix) else np.nan,
    }
    return res, true_err, est_err


def print_summary(r):
    print(f"[{r['scenario']}] final state {r['final_state']}, robot says docked: {r['docked']}, time {r['time']:.1f} s "
          f"(DOCK from {r['dock_start']:.1f} s, paused {r['paused']:.1f} s, ACQUIRE {r['acquire']:.1f} s)")
    print(f"  tracking (true): lateral RMSE {r['lateral_rmse']:.3f} m, max {r['lateral_max']:.3f} m, heading RMSE {r['heading_rmse']:.2f} deg")
    print(f"  estimation: position RMSE {r['est_rmse']:.3f} m, max {r['est_max']:.3f} m, mean NEES {r['nees']:.2f} (after first marker fix {r['nees_marker']:.2f})")
    if r["bad_pnp"]:
        print(f"  dropped {len(r['bad_pnp'])} non-finite PnP results")
    if r["relocalised"]:
        print(f"  relocalised at t = {', '.join(f'{x:.1f}' for x in r['relocalised'])} s")
    print(f"  docking error (true): {r['dock_pos']:.3f} m, heading {r['dock_heading']:+.2f} deg, min front clearance {r['min_clear']:.3f} m")
    if r["false_ids"]:
        print(f"  ignored {len(r['false_ids'])} detections of unknown marker ids {sorted({i for _, i in r['false_ids']})}")


def plot(log, truth, true_err, est_err, out):
    t, traj = log["t"], log["traj"]
    sig = np.sqrt(np.array([np.diag(P) for P in log["P"]]))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"scenario: {log['scenario']}")

    # path
    a = ax[0, 0]
    for x0, y0, w, h in [(-0.8, -0.9, 0.4, 0.8), (0.5, -1.2, 0.6, 0.4), (1.9, 1.0, 0.1, 0.4)]:
        a.add_patch(mpatches.Rectangle((x0, y0), w, h, color="0.6"))
    for cx, cy, r in [(0.2, 0.6, 0.2), (-1.0, 1.0, 0.15)]:
        a.add_patch(mpatches.Circle((cx, cy), r, color="0.6"))
    a.plot(*traj.xy_path.T, "k--", lw=1, label="reference")
    a.plot(*truth[:, :2].T, "g", label="true")
    a.plot(*log["est"][:, :2].T, "b:", label="EKF estimate")
    if "push" in log["scenario"] or log["scenario"] == "all":
        i = np.searchsorted(t, PUSH_TIME)
        a.plot(*truth[i, :2], "mo", label="push")
    blind = log["blind"]
    if np.any(blind):
        a.plot(*truth[blind, :2].T, color="orange", lw=3, alpha=0.6, label="camera blind")
    a.plot(*DOCK_POSE[:2], "r*", ms=10, label="dock point")
    a.set_aspect("equal"); a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.legend(fontsize=7, loc="upper left"); a.set_title("path")

    # tracking errors
    for a, i, name, k in [(ax[0, 1], 1, "lateral error (m)", 1), (ax[0, 2], 2, "heading error (deg)", np.rad2deg(1))]:
        a.plot(t, true_err[:, i] * k, "g", label="true vs reference")
        a.plot(t, log["err"][:, i] * k, "b:", label="estimate vs reference")
        a.set_title(name); a.set_xlabel("t (s)")
    ax[0, 1].legend(fontsize=8)

    # estimation error
    a = ax[1, 0]
    a.plot(t, np.hypot(est_err[:, 0], est_err[:, 1]), "b", label="position error")
    a.plot(t, 3 * np.hypot(sig[:, 0], sig[:, 1]), "b", alpha=0.3, label="3σ")
    a.set_title("estimation error (m)"); a.set_xlabel("t (s)"); a.legend(fontsize=8)

    # commands
    a = ax[1, 1]
    a.plot(t, log["cmd"][:, 0], "b", label="v (m/s)")
    a.plot(t, log["cmd"][:, 1], "r", label="w (rad/s)")
    a.set_title("commands"); a.set_xlabel("t (s)"); a.legend(fontsize=8)

    # states and flags
    a = ax[1, 2]
    levels = {"NAVIGATE": 0, "DOCK": 1, "ACQUIRE": 2, "DONE": 3, "ABORT": 4}
    a.step(t, [levels[s] for s in log["state"]], "k", where="post", label="state")
    a.fill_between(t, -0.5, 4.5, where=log["blind"], color="orange", alpha=0.3, label="camera blind")
    a.fill_between(t, -0.5, 4.5, where=log["paused"], color="r", alpha=0.3, label="paused")
    a.set_yticks(list(levels.values()), list(levels.keys()))
    a.set_title("state machine"); a.set_xlabel("t (s)"); a.legend(fontsize=8, loc="center left")

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"  saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-error", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    rec = TruthRecorder()
    log = run(args.scenario, args.seed, args.init_error, args.viewer, observer=rec)
    truth = np.array(rec.poses)
    res, true_err, est_err = summarise(log, truth)
    print_summary(res)
    if not args.no_plot:
        plot(log, truth, true_err, est_err, ROOT / "docs" / f"run_{args.scenario}.png")
