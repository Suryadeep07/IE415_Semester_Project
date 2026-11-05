import numpy as np
from .common import rot_err_deg


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, md - R @ ms


def trajectory_metrics(poses, gt, align=False):
    P = np.array([p[:3, 3] for p in poses])
    G = np.array([g[:3, 3] for g in gt])
    if align:
        R, t = umeyama(P, G)
        P = (R @ P.T).T + t
    e = np.linalg.norm(P - G, axis=1)          # e_p(k) = ||p_est(k) - p_gt(k)||
    L = float(np.sum(np.linalg.norm(np.diff(G, axis=0), axis=1)))
    m = dict(rmse_m=float(np.sqrt(np.mean(e ** 2))), max_m=float(e.max()),
             endpoint_drift_m=float(e[-1]), drift_per_m=float(e[-1] / L), path_length_m=L)
    if not align:
        m["mean_heading_err_deg"] = float(np.mean([rot_err_deg(a[:3, :3], b[:3, :3])
                                                   for a, b in zip(poses, gt)]))
    return m, e


def revisit_gap(poses, gt, radius=0.15, min_sep=30):
    P = np.array([p[:3, 3] for p in poses])
    G = np.array([g[:3, 3] for g in gt])
    gaps = [np.linalg.norm(P[i] - P[j]) for i in range(len(G)) for j in range(i + min_sep, len(G))
            if np.linalg.norm(G[i] - G[j]) < radius]
    return float(np.mean(gaps)) if gaps else None


def marker_errors(est, gt):
    errs = {int(i): float(np.linalg.norm(est[i][:3, 3] - gt[i][:3, 3])) for i in est if i in gt}
    return dict(mean_m=float(np.mean(list(errs.values()))), max_m=float(max(errs.values())),
                per_marker=errs) if errs else {}


def marker_pair_errors(est, tape):
    out = {}
    for key, d in tape.items():
        a, b = (int(x) for x in key.split('-'))
        if a in est and b in est:
            out[key] = float(np.linalg.norm(est[a][:3, 3] - est[b][:3, 3]) - d)
    if not out:
        return {}
    v = np.abs(list(out.values()))
    return dict(mean_abs_m=float(v.mean()), max_abs_m=float(v.max()), per_pair=out)
