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
from run import TruthRecorder, summarise
from src.pipeline import run
from src.scenarios import SCENARIOS

SUCCESS_POS = 0.03
SUCCESS_HEADING = 3.0
FIELDS = ["scenario", "seed", "final_state", "docked", "success", "time", "acquire", "relocalisations",
          "lateral_rmse", "lateral_max", "heading_rmse", "est_rmse", "est_max", "nees_marker",
          "dock_pos", "dock_heading", "min_clear", "bias", "bias_wheel", "push_time", "push_dir", "blind_start"]


# one randomised trial
def trial(job):
    scenario, seed = job
    rec = TruthRecorder()
    log = run(scenario, seed, init_error=True, jitter=True, observer=rec)
    res, _, _ = summarise(log, np.array(rec.poses))
    res.update(log["settings"])
    res["seed"] = seed
    res["relocalisations"] = len(res["relocalised"])
    res["success"] = res["final_state"] == "DONE" and res["dock_pos"] < SUCCESS_POS and abs(res["dock_heading"]) < SUCCESS_HEADING
    print(f"{scenario:9s} seed {seed:2d}: {res['final_state']:5s} dock {res['dock_pos'] * 1000:5.1f} mm {res['dock_heading']:+5.2f} deg", flush=True)
    return {k: res[k] for k in FIELDS}


def table(rows):
    print(f"\n{'scenario':10s} {'success':>8s} {'dock error mm':>16s} {'heading deg':>13s} {'lateral RMSE mm':>16s} {'est RMSE mm':>13s} {'NEES median':>12s} {'reloc':>6s}")
    for sc in SCENARIOS:
        r = [x for x in rows if x["scenario"] == sc]
        if not r:
            continue
        g = lambda k: np.array([x[k] for x in r], dtype=float)
        ok = g("success")
        print(f"{sc:10s} {int(ok.sum()):3d}/{len(r):<4d} {g('dock_pos').mean() * 1000:7.1f} ± {g('dock_pos').std() * 1000:5.1f} "
              f"{np.abs(g('dock_heading')).mean():6.2f} ± {np.abs(g('dock_heading')).std():4.2f} "
              f"{g('lateral_rmse').mean() * 1000:7.1f} ± {g('lateral_rmse').std() * 1000:5.1f} "
              f"{g('est_rmse').mean() * 1000:6.1f} ± {g('est_rmse').std() * 1000:4.1f} {np.median(g('nees_marker')):12.2f} {g('relocalisations').mean():6.1f}")


def plot(rows, out):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    names = [s for s in SCENARIOS if any(x["scenario"] == s for x in rows)]
    for a, key, scale, label in [(ax[0], "dock_pos", 1000, "docking error (mm)"), (ax[1], "lateral_rmse", 1000, "lateral tracking RMSE (mm)"),
                                 (ax[2], "est_rmse", 1000, "estimation RMSE (mm)"), (ax[3], "nees_marker", 1, "mean NEES after first fix")]:
        data = [[x[key] * scale for x in rows if x["scenario"] == s] for s in names]
        a.boxplot(data, showfliers=False)
        for i, d in enumerate(data):
            a.plot(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d, "o", ms=3, alpha=0.6)
        a.set_xticks(range(1, len(names) + 1), names, rotation=30)
        a.set_title(label)
    ax[3].axhline(3, color="g", ls=":", label="ideal 3")
    ax[3].set_yscale("log")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", default=str(ROOT / "analysis" / "results" / "trials.csv"))
    args = p.parse_args()

    jobs = [(s, seed) for s in args.scenarios for seed in range(args.trials)]
    with Pool(args.jobs) as pool:
        rows = pool.map(trial, jobs, chunksize=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")
    table(rows)
    plot(rows, ROOT / "docs" / "trials.png")
