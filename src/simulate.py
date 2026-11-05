import numpy as np
from scipy.spatial.transform import Rotation as Rot
from .common import make_T, inv_T

Z = 1.4  # camera and marker height (m)


def marker_poses():
    spec = [((1.5, 0), (0, 1)), ((4.5, 0), (0, 1)), ((6, 1), (-1, 0)), ((6, 3), (-1, 0)),
            ((4.5, 4), (0, -1)), ((1.5, 4), (0, -1)), ((0, 3), (1, 0)), ((0, 1), (1, 0))]
    out = {}
    for i, ((x, y), (nx, ny)) in enumerate(spec):
        zm, ym = np.array([nx, ny, 0.]), np.array([0, 0, 1.])  # z = out of wall, y = up
        out[i] = make_T(np.column_stack([np.cross(ym, zm), ym, zm]), [x, y, Z])
    return out


def _cam_R(f):
    f = f / np.linalg.norm(f)
    right = np.cross([0, 0, -1.], f)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(f, right), f])


def make_gt(ds=0.12):
    wp = np.array([[1, 1], [5, 1], [5, 3], [1, 3], [1, 1], [5, 1], [5, 2]], float)
    seg = np.diff(wp, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(L)]
    pts, hd = [], []
    for s in np.arange(0, cum[-1], ds):
        j = min(np.searchsorted(cum, s, side='right') - 1, len(L) - 1)
        u = seg[j] / L[j]
        pts.append(wp[j] + u * (s - cum[j]))
        hd.append(np.arctan2(u[1], u[0]))
    w = np.ones(9) / 9
    hs = np.arctan2(np.convolve(np.sin(hd), w, 'same'), np.convolve(np.cos(hd), w, 'same'))
    gt = []
    for k, (p, h) in enumerate(zip(pts, hs)):
        a = h - np.pi / 2 + 0.6 * np.sin(2 * np.pi * k * ds / 6.0)  # look at the right-hand wall, sweeping
        gt.append(make_T(_cam_R(np.array([np.cos(a), np.sin(a), 0.])), [p[0], p[1], Z]))
    return gt


def noisy_odom(gt, rng, rot_sig=0.004, yaw_bias=0.002, scale=1.02, t_sig=0.003):
    odom = [np.eye(4)]  # odometry frame starts at identity (unknown relation to the room)
    for k in range(len(gt) - 1):
        rel = inv_T(gt[k]) @ gt[k + 1]
        dR = Rot.from_rotvec(np.array([0, yaw_bias, 0]) + rng.normal(0, rot_sig, 3)).as_matrix()
        odom.append(odom[-1] @ make_T(rel[:3, :3] @ dR, rel[:3, 3] * scale + rng.normal(0, t_sig, 3)))
    return odom


def observe(gt, markers, rng, loss=(55, 75), p_out=0.03, p_miss=0.1):
    obs = []
    for k, T in enumerate(gt):
        if loss[0] <= k < loss[1]:
            continue  # marker-loss segment
        for i, M in markers.items():
            Tcm = inv_T(T) @ M
            p = Tcm[:3, 3]
            d = np.linalg.norm(p)
            if p[2] < 0.3 or d > 3.5 or abs(p[0] / p[2]) > 0.7 or abs(p[1] / p[2]) > 0.5:
                continue
            if Tcm[:3, 2] @ (-p / d) < 0.42 or rng.random() < p_miss:
                continue
            sc = max(1.0, d)
            if rng.random() < p_out:  # false / bad detection
                noise = make_T(Rot.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix(), rng.normal(0, 0.3, 3))
            else:
                noise = make_T(Rot.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix(), rng.normal(0, 0.01 * sc, 3))
            obs.append((k, i, Tcm @ noise))
    return obs


def simulate(seed=0, loss=(55, 75), p_out=0.03, **kw):
    rng = np.random.default_rng(seed)
    gt, markers = make_gt(), marker_poses()
    return dict(gt=gt, odom=noisy_odom(gt, rng), obs=observe(gt, markers, rng, loss, p_out),
                markers_gt=markers, dt=0.24, loss=loss)
