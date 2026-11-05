import numpy as np
from scipy.spatial.transform import Rotation as Rot


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def T_from_rvec_tvec(rvec, tvec):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return make_T(R, np.ravel(tvec))


def T_from_xyz_quat(x, y, z, qx, qy, qz, qw):
    return make_T(Rot.from_quat([qx, qy, qz, qw]).as_matrix(), [x, y, z])
