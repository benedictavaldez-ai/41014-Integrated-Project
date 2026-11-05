# 41014 Integrated Project

Track A: mobile robot visual waypoint navigation. A TurtleBot3 Waffle Pi in MuJoCo navigates a room with obstacles using its 2D LiDAR and docks at an ArUco-marked charging station. The pipeline is a state-space model of the platform, an EKF fusing wheel-encoder odometry with camera measurements of the marker, and an LQR trajectory tracker that uses only the estimated state.

**Project page:** https://benedictavaldez-ai.github.io/41014-Integrated-Project/

![Simulation world](docs/world_overview.png)

## Group Information

Project number: 38

| Name | Student ID |
|---|---|
| Benedict Valdez | 25470271 |
| Jacob Jung | 25469949 |

## Install Info

Tested with Python 3.10 or newer.

```
git clone https://github.com/benedictavaldez-ai/41014-Integrated-Project.git
cd 41014-Integrated-Project
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

Then install the dependencies:

```
pip install -r requirements.txt
```

## Run Commands

Sensor smoke test with the MuJoCo viewer (drives an open-loop path and prints encoder, LiDAR and ArUco readings):

```
python main.py
```

Same test without the viewer:

```
python main.py --headless
```

Controllability and observability check:

```
python analysis/model_check.py
```

Marker measurement noise study (writes `docs/marker_noise.png`):

```
python analysis/marker_noise.py
```

EKF test on the open-loop path (writes `docs/ekf_open_loop.png`):

```
python analysis/ekf_open_loop.py
```

Same test with a random initial estimate error (writes `docs/ekf_open_loop_init_seed3.png`):

```
python analysis/ekf_open_loop.py --init-error --seed 3
```

Closed-loop test, EKF + LQR from start to dock (writes `docs/tracking.png`):

```
python analysis/tracking_test.py
```

Same test with a random initial estimate error (writes `docs/tracking_init_seed3.png`):

```
python analysis/tracking_test.py --init-error --seed 3
```

Full pipeline from start to dock, scored against ground truth (writes `docs/run_<scenario>.png`):

```
python run.py --scenario nominal --viewer
```

Scenarios: `nominal`, `odom_bias`, `push`, `occlusion`, `all`. `--viewer` opens the MuJoCo viewer in real time, `--init-error` starts with a wrong estimate, `--seed` sets the random seed.

Repeated randomised trials (writes `analysis/results/trials.csv` and `docs/trials.png`):

```
python analysis/trials.py --trials 10 --jobs 2
```

LQR and EKF trade-off sweeps (writes `analysis/results/sweeps.csv`, `docs/sweep_lqr.png`, `docs/sweep_ekf.png`; `--plot-only` redraws the plots from the saved CSV):

```
python analysis/sweeps.py --jobs 2
```

Each run renders the camera at 10 Hz and takes about 1 to 2 minutes on a CPU; `--jobs` sets how many runs go in parallel.

## Repository Structure

```
models/
  turtlebot3/        unmodified ROBOTIS TurtleBot3 MJCF and meshes (Apache 2.0)
  tb3_robot.xml      Waffle Pi with camera, LiDAR site and wheel encoders added
  world.xml          room, obstacles, charging dock with ArUco marker
  textures/          ArUco marker image (DICT_4X4_50, id 0)
src/
  perception/        LiDAR ray-cast scanner, ArUco detection and pose,
                     wheel encoders, marker measurement and noise model
  estimation/        EKF
  control/           reference trajectory, LQR tracker, LiDAR speed limit
  pipeline.py        full robot loop and state machine
  scenarios.py       disturbances: odometry bias, push, camera occlusion
analysis/            model checks, sensor calibration, EKF and tracking tests,
                     repeated trials and trade-off sweeps
  results/           trial and sweep results (CSV)
docs/                figures and the project page
run.py               entry point: run a scenario, score it against ground truth, plot it
main.py              sensor smoke test
```

Simulator ground truth is read only in `analysis/` and in the evaluation-only observer in `run.py`; nothing in `src/` uses it.

## Acknowledgements

- TurtleBot3 Waffle Pi MJCF model and meshes: ROBOTIS, Apache 2.0 (`models/turtlebot3/`, unmodified; changes are in `models/tb3_robot.xml`).
- Simulation: MuJoCo. Marker detection and pose: OpenCV ArUco and `solvePnP` (IPPE_SQUARE).
- The discrete Riccati equation is solved with the structure-preserving doubling algorithm, checked against SciPy's `solve_discrete_are`.
- Project page template: [Nerfies](https://github.com/nerfies/nerfies.github.io), CC BY-SA 4.0.
