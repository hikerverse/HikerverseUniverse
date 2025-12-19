# python
import numpy as np

def _normalize_rows(v):
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def estimate_attitude(inertial_vecs, body_vecs, weights=None):
    """
    Solve Wahba's problem (Kabsch/SVD):
    Find R that minimizes sum_i w_i * || b_i - R @ r_i ||^2
    where r_i are inertial unit vectors and b_i are measured body-frame unit vectors.
    Returns rotation matrix R (3x3) such that b ≈ R @ r.
    """
    r = _normalize_rows(inertial_vecs)
    b = _normalize_rows(body_vecs)
    if r.shape != b.shape or r.shape[1] != 3:
        raise ValueError("Input vectors must be N x 3 and match in count")

    n = r.shape[0]
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != n:
            raise ValueError("weights length must match number of vectors")

    # cross-covariance
    B = (b * w[:, None]).T @ r  # 3x3
    U, S, Vt = np.linalg.svd(B)
    M = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        M[2, 2] = -1  # ensure proper rotation (det = +1)
    R = U @ M @ Vt
    return R

def rot_to_quat(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def rot_to_euler_zyx(R, degrees=True):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll) in Z-Y-X convention:
    yaw (psi) around Z, pitch (theta) around Y, roll (phi) around X.
    """
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy < 1e-8:
        # Gimbal lock: set yaw = 0
        yaw = 0.0
        pitch = np.arctan2(sy, cy)
        roll = np.arctan2(-R[0, 1], R[1, 1])
    else:
        pitch = np.arctan2(sy, cy)
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    if degrees:
        return np.degrees([yaw, pitch, roll])
    return np.array([yaw, pitch, roll])

def rotation_angle_between(R1, R2):
    """Angle (radians) of rotation that maps R1 to R2: R_delta = R2 @ R1.T"""
    R_delta = R2 @ R1.T
    trace = np.clip(np.trace(R_delta), -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return angle


if __name__ == "__main__":
    # Example: synthetic test
    rng = np.random.default_rng(42)

    # create a "true" attitude (rotation) by axis-angle
    axis = np.array([0.3, 0.7, 0.65])
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(30.0)  # 30 degrees rotation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_true = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

    # generate some inertial star unit vectors (e.g., catalog directions)
    n_stars = 10
    # random directions uniform on sphere
    u = rng.normal(size=(n_stars, 3))
    u = _normalize_rows(u)

    # body measurements: apply true rotation and add small noise
    noise_level = 0.005  # radians ~ sub-degree directional noise
    noise = rng.normal(scale=noise_level, size=(n_stars, 3))
    b_meas = (R_true @ u.T).T + noise
    b_meas = _normalize_rows(b_meas)

    # estimate attitude
    R_est = estimate_attitude(u, b_meas)
    q_est = rot_to_quat(R_est)
    yaw, pitch, roll = rot_to_euler_zyx(R_est)

    ang_err_deg = np.degrees(rotation_angle_between(R_true, R_est))

    print(f"Estimated quaternion (w,x,y,z): {q_est}")
    print(f"Estimated yaw,pitch,roll (deg): {yaw:.3f}, {pitch:.3f}, {roll:.3f}")
    print(f"Angular error between true and estimated attitude: {ang_err_deg:.4f} deg")
