import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
from .common import make_T, inv_T


def _marker_scale(a, b):
    if not a or not b:
        return None
    s = [np.linalg.norm((a[i] @ inv_T(b[i]))[:3, 3]) for i in set(a) & set(b)]
    s = float(np.median(s)) if s else None
    return s if s and 0.002 < s < 1.0 else None


def _load_imu_gyro(imu_csv):
    if not imu_csv:
        return None
    try:
        df = pd.read_csv(imu_csv)
        if {"t", "gx", "gy", "gz"}.issubset(df.columns):
            return df["t"].values, df[["gx", "gy", "gz"]].values
    except Exception:
        pass
    return None


def _gyro_dR(imu_data, t1, t2):
    if imu_data is None or t2 <= t1:
        return None
    t, gyr = imu_data
    i0, i1 = np.searchsorted(t, t1), np.searchsorted(t, t2)
    if i1 <= i0:
        return None
    R = Rot.identity()
    for k in range(i0, min(i1, len(t) - 1)):
        dt = float(t[k + 1] - t[k])
        if 0.0 < dt < 0.5:
            R = R * Rot.from_rotvec(gyr[k] * dt)
    return R.as_matrix()


def run_vo(video, K, dist, marker_obs, step=3, default_scale=0.05, imu_csv=None):
    """marker_obs: {frame: {id: T_C_M}}. Returns (frame_indices, poses T_W_C in odometry frame)."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    imu_data = _load_imu_gyro(imu_csv)
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev, idxs, poses, T, scale, f = None, [], [], np.eye(4), default_scale, -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if f % step:
            continue
        t_cur = f / fps
        kp, des = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if prev is not None:
            dR_imu = _gyro_dR(imu_data, prev["t"], t_cur)
            moved = False
            if des is not None and prev["des"] is not None:
                m = bf.match(prev["des"], des)
                if len(m) >= 25:
                    p1 = np.float32([prev["kp"][x.queryIdx].pt for x in m])
                    p2 = np.float32([kp[x.trainIdx].pt for x in m])
                    flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
                    if flow >= 1.5:  # Gate out stationary camera noise
                        p1_u = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
                        p2_u = cv2.undistortPoints(p2.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
                        E, mask = cv2.findEssentialMat(p1_u, p2_u, K, cv2.RANSAC, 0.999, 1.0)
                        if E is not None and E.shape == (3, 3):
                            _, R, t, _ = cv2.recoverPose(E, p1_u, p2_u, K, mask=mask)
                            s = _marker_scale(marker_obs.get(prev["f"]), marker_obs.get(f))
                            if s:
                                scale = 0.7 * scale + 0.3 * s
                            step_T = inv_T(make_T(R, t.ravel() * scale))
                            if dR_imu is not None:
                                step_T[:3, :3] = dR_imu
                            T = T @ step_T
                            moved = True
            if not moved and dR_imu is not None:
                T = T @ make_T(dR_imu, [0.0, 0.0, 0.0])
        idxs.append(f)
        poses.append(T.copy())
        prev = dict(kp=kp, des=des, f=f, t=t_cur)
    cap.release()
    return idxs, poses