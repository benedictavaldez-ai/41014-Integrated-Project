import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.estimation.ekf as ekf_mod
import src.perception.marker_pose as mp_mod
from run import TruthRecorder, summarise
from src.control.lqr import Q as LQR_Q, R as LQR_R
from src.pipeline import run, wheel_speeds

# sweep values
LQR_R_SCALE = [0.25, 0.5, 1.0, 2.0, 4.0]
R_INFLATION = [1.0, 2.0, 3.0, 5.0]
WHEEL_NOISE = [0.01, 0.02, 0.05, 0.10]
SEEDS = [0, 1, 2]
DEFAULTS = {"inflation": mp_mod.INFLATION, "wheel_noise": ekf_mod.WHEEL_NOISE_K, "r_scale": 1.0}
CONVERGED = 0.02


# one run with the given settings
def job(spec):
    sweep, scenario, seed, value = spec
    cfg = dict(DEFAULTS)
    cfg[{"lqr": "r_scale", "inflation": "inflation", "wheel_noise": "wheel_noise"}[sweep]] = value
    mp_mod.INFLATION = cfg["inflation"]
    ekf_mod.WHEEL_NOISE_K = cfg["wheel_noise"]

    rec = TruthRecorder()
    log = run(scenario, seed, init_error=True, lqr_weights=(LQR_Q, LQR_R * cfg["r_scale"]), observer=rec)
    truth = np.array(rec.poses)
    res, _, est_err = summarise(log, truth)

    # control effort
    moving = log["ref_clock"] < log["traj"].duration
    dv = log["cmd"][moving, 0] - log["ref"][moving, 3]
    dw = log["cmd"][moving, 1] - log["ref"][moving, 4]
    wheels = np.array([wheel_speeds(v, w) for v, w in log["cmd"]])

    # estimate convergence
    pos_err = np.hypot(est_err[:, 0], est_err[:, 1])
    fix = np.argmax(log["marker_ok"]) if np.any(log["marker_ok"]) else None
    conv = np.nan
    if fix is not None:
        below = np.where(pos_err[fix:] < CONVERGED)[0]
        conv = log["t"][fix + below[0]] - log["t"][fix] if len(below) else np.nan

    out = {
        "sweep": sweep, "scenario": scenario, "seed": seed, "value": value,
        "success": res["final_state"] == "DONE" and res["dock_pos"] < 0.03,
        "dock_pos": res["dock_pos"], "lateral_rmse": res["lateral_rmse"], "lateral_max": res["lateral_max"],
        "heading_rmse": res["heading_rmse"], "dv_rms": np.sqrt(np.mean(dv ** 2)), "dw_rms": np.sqrt(np.mean(dw ** 2)),
        "wheel_peak": np.abs(wheels).max(), "est_rmse": res["est_rmse"], "est_max": res["est_max"],
        "nees_marker": res["nees_marker"], "converge_time": conv, "relocalisations": len(res["relocalised"]),
    }
    print(f"{sweep:11s} {value:5.2f} seed {seed}: lateral {out['lateral_rmse'] * 1000:5.1f} mm, dw {out['dw_rms']:.3f}, "
          f"est {out['est_rmse'] * 1000:5.1f} mm, NEES {out['nees_marker']:6.2f}, dock {out['dock_pos'] * 1000:4.1f} mm", flush=True)
    return out


def mean_by(rows, sweep, key):
    vals = sorted({r["value"] for r in rows if r["sweep"] == sweep})
    m = [np.nanmean([r[key] for r in rows if r["sweep"] == sweep and r["value"] == v]) for v in vals]
    s = [np.nanstd([r[key] for r in rows if r["sweep"] == sweep and r["value"] == v]) for v in vals]
    return np.array(vals), np.array(m), np.array(s)


def report(rows):
    for sweep, keys in [("lqr", ["lateral_rmse", "lateral_max", "dw_rms", "dv_rms", "wheel_peak", "dock_pos"]),
                        ("inflation", ["nees_marker", "est_rmse", "est_max", "converge_time", "dock_pos"]),
                        ("wheel_noise", ["nees_marker", "est_rmse", "est_max", "relocalisations", "dock_pos"])]:
        if not any(r["sweep"] == sweep for r in rows):
            continue
        print(f"\n{sweep}: " + "  ".join(keys))
        vals = mean_by(rows, sweep, keys[0])[0]
        for i, v in enumerate(vals):
            cells = [mean_by(rows, sweep, k) for k in keys]
            ok = sum(r["success"] for r in rows if r["sweep"] == sweep and r["value"] == v)
            n = sum(1 for r in rows if r["sweep"] == sweep and r["value"] == v)
            print(f"  {v:5.2f}  " + "  ".join(f"{c[1][i]:.4f}±{c[2][i]:.4f}" for c in cells) + f"  success {ok}/{n}")


def plot(rows):
    if any(r["sweep"] == "lqr" for r in rows):
        v, dock, dock_s = mean_by(rows, "lqr", "dock_pos")
        _, dv, dv_s = mean_by(rows, "lqr", "dv_rms")
        _, lat, lat_s = mean_by(rows, "lqr", "lateral_rmse")
        _, dw, dw_s = mean_by(rows, "lqr", "dw_rms")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].errorbar(dv, dock * 1000, xerr=dv_s, yerr=dock_s * 1000, fmt="o-", capsize=3)
        for x, y, k in zip(dv, dock * 1000, v):
            ax[0].annotate(f"R × {k:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax[0].set_xlabel("RMS speed correction (m/s)")
        ax[0].set_ylabel("docking error (mm)")
        ax[0].set_title("effort vs docking accuracy")
        ax[1].errorbar(dw, lat * 1000, xerr=dw_s, yerr=lat_s * 1000, fmt="o-", capsize=3)
        for x, y, k in zip(dw, lat * 1000, v):
            ax[1].annotate(f"R × {k:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax[1].set_xlabel("RMS turn-rate correction (rad/s)")
        ax[1].set_ylabel("lateral tracking RMSE, true (mm)")
        ax[1].set_title("effort vs lateral tracking")
        fig.suptitle("LQR R sweep, push scenario (3 seeds each)")
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / "sweep_lqr.png", dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, sweep, xlabel in [(ax[0], "inflation", "R inflation factor (nominal)"), (ax[1], "wheel_noise", "wheel noise k in Q (odometry bias)")]:
        if not any(r["sweep"] == sweep for r in rows):
            continue
        v, nees, nees_s = mean_by(rows, sweep, "nees_marker")
        _, est, est_s = mean_by(rows, sweep, "est_rmse")
        a.errorbar(v, nees, yerr=nees_s, fmt="o-", color="tab:blue", capsize=3, label="mean NEES after first fix")
        a.axhline(3, color="tab:blue", ls=":", lw=1)
        a.set_yscale("log")
        a.set_xlabel(xlabel)
        a.set_ylabel("NEES (ideal 3)", color="tab:blue")
        b = a.twinx()
        b.errorbar(v, est * 1000, yerr=est_s * 1000, fmt="s--", color="tab:red", capsize=3, label="estimation RMSE")
        b.set_ylabel("estimation RMSE (mm)", color="tab:red")
        a.set_title(f"EKF trade-off: {sweep.replace('_', ' ')} (3 seeds each)")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "sweep_ekf.png", dpi=110)
    print("saved docs/sweep_lqr.png, docs/sweep_ekf.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs="+", default=["lqr", "inflation", "wheel_noise"], choices=["lqr", "inflation", "wheel_noise"])
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", default=str(ROOT / "analysis" / "results" / "sweeps.csv"))
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    # redraw from saved results
    if args.plot_only:
        with open(args.out) as f:
            rows = [{k: (v if k in ("sweep", "scenario") else v == "True" if v in ("True", "False") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        report(rows)
        plot(rows)
        sys.exit()

    specs = []
    if "lqr" in args.sweeps:
        specs += [("lqr", "push", s, v) for v in LQR_R_SCALE for s in SEEDS]
    if "inflation" in args.sweeps:
        specs += [("inflation", "nominal", s, v) for v in R_INFLATION for s in SEEDS]
    if "wheel_noise" in args.sweeps:
        specs += [("wheel_noise", "odom_bias", s, v) for v in WHEEL_NOISE for s in SEEDS]
    with Pool(args.jobs) as pool:
        rows = pool.map(job, specs, chunksize=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")
    report(rows)
    plot(rows)
