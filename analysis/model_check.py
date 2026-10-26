import numpy as np

DT = 0.05
CAMERA_OFFSET = 0.086
LANDMARK = np.array([1.899, 1.2])


def tracking_error_model(v_ref, w_ref):
    A = np.array([[0.0, w_ref, 0.0], [-w_ref, 0.0, v_ref], [0.0, 0.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return A, B


def controllability_rank(A, B):
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    return np.linalg.matrix_rank(C, tol=1e-9)


def motion_jacobian(x, v):
    return np.array([[1.0, 0.0, -v * np.sin(x[2]) * DT], [0.0, 1.0, v * np.cos(x[2]) * DT], [0.0, 0.0, 1.0]])


def step(x, v, w):
    return x + DT * np.array([v * np.cos(x[2]), v * np.sin(x[2]), w])


def range_bearing_jacobian(x):
    cx = x[0] + CAMERA_OFFSET * np.cos(x[2])
    cy = x[1] + CAMERA_OFFSET * np.sin(x[2])
    dx, dy = LANDMARK[0] - cx, LANDMARK[1] - cy
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    dcx_dth = -CAMERA_OFFSET * np.sin(x[2])
    dcy_dth = CAMERA_OFFSET * np.cos(x[2])
    H_range = np.array([-dx / r, -dy / r, -(dx * dcx_dth + dy * dcy_dth) / r])
    H_bearing = np.array([dy / q, -dx / q, (dy * dcx_dth - dx * dcy_dth) / q - 1.0])
    return np.vstack([H_range, H_bearing])


def observability_rank(x0, v, w, steps, measurement):
    x = x0.copy()
    Phi = np.eye(3)
    rows = []
    for _ in range(steps):
        H = range_bearing_jacobian(x) if measurement == "range_bearing" else np.eye(3)
        rows.append(H @ Phi)
        Phi = motion_jacobian(x, v) @ Phi
        x = step(x, v, w)
    return np.linalg.matrix_rank(np.vstack(rows), tol=1e-9)


if __name__ == "__main__":
    print("Controllability of the tracking-error model (rank out of 3)")
    for v_ref, w_ref in [(0.2, 0.0), (0.2, 0.5), (0.0, 0.5), (0.0, 0.0)]:
        A, B = tracking_error_model(v_ref, w_ref)
        print(f"  v_ref={v_ref:.1f} m/s, w_ref={w_ref:.1f} rad/s -> rank {controllability_rank(A, B)}")

    x0 = np.array([0.0, 0.0, 0.5])
    print("Observability with the ArUco marker (rank out of 3)")
    print(f"  range+bearing, one instant           -> rank {observability_rank(x0, 0.0, 0.0, 1, 'range_bearing')}")
    print(f"  range+bearing, robot stationary 1 s  -> rank {observability_rank(x0, 0.0, 0.0, 20, 'range_bearing')}")
    print(f"  range+bearing, robot moving 1 s      -> rank {observability_rank(x0, 0.2, 0.0, 20, 'range_bearing')}")
    print(f"  full marker pose (PnP), one instant  -> rank {observability_rank(x0, 0.0, 0.0, 1, 'pose')}")
