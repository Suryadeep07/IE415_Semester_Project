import cv2
import numpy as np
from .common import T_from_rvec_tvec


def make_detector(name="DICT_4X4_50"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def detect_poses(img, det, K, dist, size, max_reproj=2.0, min_cos=0.3):
    corners, ids, _ = det.detectMarkers(img)
    out = []
    if ids is None:
        return out
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    for c, i in zip(corners, ids.ravel()):
        pts = c.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - pts, axis=1)))
        T = T_from_rvec_tvec(rvec, tvec)
        cosang = float(T[:3, 2] @ (-T[:3, 3] / np.linalg.norm(T[:3, 3])))
        if err > max_reproj or cosang < min_cos:
            continue
        out.append(dict(id=int(i), T=T, err=err, rvec=rvec, tvec=tvec))
    return out
