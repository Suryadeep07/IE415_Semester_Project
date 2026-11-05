"""IMU-only dead reckoning baseline (gyro -> orientation, accel -> velocity -> position).
imu.csv columns: t (s), gx, gy, gz (rad/s), ax, ay, az (m/s^2, still phone reads ~ +9.81 m'up')."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot


def dead_reckon(csv, g=9.81):
    df = pd.read_csv(csv)
    t = df["t"].values
    gyr, acc = df[["gx", "gy", "gz"]].values, df[["ax", "ay", "az"]].values
    n0 = max(5, int(np.searchsorted(t, t[0] + 0.5)))
    up = acc[:n0].mean(0)
    R, _ = Rot.align_vectors([[0, 0, 1]], [up / np.linalg.norm(up)])
    v, p, P = np.zeros(3), np.zeros(3), [np.zeros(3)]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        R = R * Rot.from_rotvec(gyr[k - 1] * dt)
        v = v + (R.apply(acc[k - 1]) - np.array([0, 0, g])) * dt
        p = p + v * dt
        P.append(p.copy())
    return t, np.array(P)
