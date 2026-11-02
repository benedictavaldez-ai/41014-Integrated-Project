import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.control.lqr import LQRTracker, tracking_error
from src.control.safety import DOCK_SLOW_DIST, DOCK_STOP_DIST, front_clearance, limit_speed
from src.control.trajectory import Trajectory
from src.estimation.ekf import EKF, WHEEL_RADIUS, WHEEL_SEPARATION, wrap
from src.perception.aruco import ArucoDetector
from src.perception.lidar import Lidar
from src.perception.marker_pose import CAM_POS, MARKER_MAP, marker_measurement
from src.scenarios import Scenario

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "world.xml"

# poses
START_POSE = np.array([-1.5, -1.5, 0.0])
DOCK_POSE = np.array([1.70, 1.20, 0.0])
P0 = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(3) ** 2])

# rates
CONTROL_PERIOD = 0.02
SENSOR_PERIOD = 0.1

# docking and marker acquisition
MARKER_TIMEOUT = 1.0
MARKER_MIN_RANGE = 0.30
DOCK_TOL_POS = 0.03
DOCK_TOL_HEADING = np.deg2rad(3)
STOP_SPEED = 0.005
STOP_HOLD = 1.0
TIME_MARGIN = 30.0
ACQUIRE_GAIN = 1.0
ACQUIRE_W_MAX = 0.3
ACQUIRE_TIMEOUT = 15.0

# relocalisation
REJECT_LIMIT = 10
P_RELOCALISE = np.diag([0.30 ** 2, 0.30 ** 2, np.deg2rad(20) ** 2])

# states
NAVIGATE, DOCK, ACQUIRE, DONE, ABORT = "NAVIGATE", "DOCK", "ACQUIRE", "DONE", "ABORT"


def wheel_speeds(v, w):
    left = (v - w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    right = (v + w * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    return left, right


# estimated range and bearing to marker
def marker_range_bearing(x, marker_id=0):
    mx, my = MARKER_MAP[marker_id][:2]
    cx = x[0] + CAM_POS[0] * np.cos(x[2])
    cy = x[1] + CAM_POS[0] * np.sin(x[2])
    return np.hypot(mx - cx, my - cy), wrap(np.arctan2(my - cy, mx - cx) - x[2])


def run(scenario="nominal", seed=0, init_error=False, viewer=False, observer=None):
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    sc = Scenario(scenario, model)
    renderer = mujoco.Renderer(model, 480, 640)
    aruco = ArucoDetector()
    lidar = Lidar(model, rng=rng)
    enc = sc.encoders(data)
    traj = Trajectory(CONTROL_PERIOD)
    tracker = LQRTracker(CONTROL_PERIOD)

    # initial estimate
    x0 = START_POSE + (rng.multivariate_normal(np.zeros(3), P0) if init_error else 0)
    ekf = EKF(x0, P0)

    log = {k: [] for k in ["t", "state", "ref_clock", "ref", "est", "P", "cmd", "err", "clear", "marker_ok", "paused", "blind"]}
    state, ref_clock = NAVIGATE, 0.0
    clearance, last_marker, still, acquire_start = np.inf, -np.inf, 0.0, 0.0
    false_ids, bad_pnp, relocalised, rejects = [], [], [], 0
    next_control, next_sensor = 0.0, 0.0
    view = mujoco.viewer.launch_passive(model, data) if viewer else None
    wall_start = time.time()

    while state not in (DONE, ABORT):
        t = data.time
        sc.apply(data, t)

        # predict
        control_tick = t >= next_control
        if control_tick:
            next_control += CONTROL_PERIOD
            dphi = enc.delta()
            ekf.predict(dphi)
            speed = WHEEL_RADIUS * abs(dphi[0] + dphi[1]) / 2 / CONTROL_PERIOD

        # sensors
        if t >= next_sensor:
            next_sensor += SENSOR_PERIOD
            angles, ranges = lidar.scan(data)
            clearance = front_clearance(angles, ranges)
            if not sc.camera_blind(t):
                renderer.update_scene(data, camera="pi_cam")
                for det in aruco.detect(renderer.render()):
                    if det["id"] not in MARKER_MAP:
                        false_ids.append((t, det["id"]))
                        continue
                    if not (np.all(np.isfinite(det["tvec"])) and np.all(np.isfinite(det["rvec"]))):
                        bad_pnp.append(t)
                        continue
                    z, R = marker_measurement(det)
                    ok_rb, _ = ekf.update_marker(z, R, det["id"])
                    if ok_rb:
                        last_marker = t
                        rejects = 0
                    else:
                        rejects += 1
                    if rejects >= REJECT_LIMIT:
                        ekf.P = ekf.P + P_RELOCALISE
                        relocalised.append(t)
                        rejects = 0

        # control
        if control_tick:
            ref = traj.sample(ref_clock)

            # state transitions
            m_range, m_bearing = marker_range_bearing(ekf.x)
            marker_ok = t - last_marker <= MARKER_TIMEOUT or m_range < MARKER_MIN_RANGE
            if state == NAVIGATE and ref_clock >= traj.dock_time:
                state = DOCK
            if state == DOCK and not marker_ok:
                state, acquire_start = ACQUIRE, t
            elif state == ACQUIRE and marker_ok:
                state = DOCK
            still = still + CONTROL_PERIOD if speed < STOP_SPEED else 0.0
            if state == DOCK and ref_clock >= traj.duration and still >= STOP_HOLD:
                state = DONE
            if t > traj.duration + TIME_MARGIN or (state == ACQUIRE and t - acquire_start > ACQUIRE_TIMEOUT):
                state = ABORT

            # command
            paused = False
            if state in (DONE, ABORT) or ref_clock >= traj.duration:
                v, w = 0.0, 0.0
                e = tracking_error(ekf.x, ref)
            elif state == ACQUIRE:
                v, w = 0.0, float(np.clip(ACQUIRE_GAIN * m_bearing, -ACQUIRE_W_MAX, ACQUIRE_W_MAX))
                e = tracking_error(ekf.x, ref)
                paused = True
            else:
                (v, w), e = tracker.command(ekf.x, ref)
                if state == DOCK:
                    v_safe = limit_speed(v, clearance, DOCK_STOP_DIST, DOCK_SLOW_DIST)
                else:
                    v_safe = limit_speed(v, clearance)
                paused = v > 0 and v_safe <= 0
                v = v_safe
                if paused:
                    v, w = 0.0, 0.0
                else:
                    ref_clock += CONTROL_PERIOD
            data.ctrl[:] = wheel_speeds(v, w)

            log["t"].append(t)
            log["state"].append(state)
            log["ref_clock"].append(ref_clock)
            log["ref"].append(ref)
            log["est"].append(ekf.x.copy())
            log["P"].append(ekf.P.copy())
            log["cmd"].append((v, w))
            log["err"].append(e)
            log["clear"].append(clearance)
            log["marker_ok"].append(marker_ok)
            log["paused"].append(paused)
            log["blind"].append(sc.camera_blind(t))
            if observer is not None:
                observer(data)

        mujoco.mj_step(model, data)
        if view is not None:
            if not view.is_running():
                break
            view.sync()
            lag = data.time - (time.time() - wall_start)
            if lag > 0:
                time.sleep(lag)

    if view is not None:
        view.close()

    # docking check
    final = ekf.x - DOCK_POSE
    final[2] = wrap(final[2])
    docked = state == DONE and np.hypot(*final[:2]) < DOCK_TOL_POS and abs(final[2]) < DOCK_TOL_HEADING

    out = {k: np.array(v) for k, v in log.items()}
    out.update(final_state=state, docked=docked, traj=traj, scenario=scenario, false_ids=false_ids, bad_pnp=bad_pnp, relocalised=relocalised)
    return out
