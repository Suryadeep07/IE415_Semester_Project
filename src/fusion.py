import numpy as np
import gtsam
from scipy.spatial.transform import Rotation as Rot
from .common import inv_T, make_T


def _X(k): return gtsam.symbol('x', int(k))
def _L(i): return gtsam.symbol('l', int(i))
def _P(T): return gtsam.Pose3(np.asarray(T, float))


def baseline_poses(odom, first_pose):
    """Odometry only: put the odometry chain into the room frame using the first pose."""
    A = first_pose @ inv_T(odom[0])
    return [A @ o for o in odom]


def _init_marker(cands):
    """Pick the candidate position closest to the median -> robust to a bad first sighting."""
    pos = np.array([c[:3, 3] for c in cands])
    med = np.median(pos, axis=0)
    return cands[int(np.argmin(np.linalg.norm(pos - med, axis=1)))]


def fuse(odom, obs, first_pose, anchors=None, sig_odom=(0.01, 0.02), sig_marker=(0.02, 0.03),
         robust=True, prior_sigma=1e-3, iters=100):
    """odom: list of 4x4 poses in ANY odometry frame. obs: list of (kf_index, marker_id, T_C_M).
    sig_* = (rotation rad, translation m). Returns dict(poses, markers, err0, err1)."""
    n = len(odom)
    base = baseline_poses(odom, first_pose)
    anchors = anchors or {}
    g, v = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for k in range(n):
        v.insert(_X(k), _P(base[k]))
    g.add(gtsam.PriorFactorPose3(_X(0), _P(first_pose),
                                 gtsam.noiseModel.Diagonal.Sigmas(np.full(6, prior_sigma))))
    on = gtsam.noiseModel.Diagonal.Sigmas(np.array([sig_odom[0]] * 3 + [sig_odom[1]] * 3))
    for k in range(n - 1):
        g.add(gtsam.BetweenFactorPose3(_X(k), _X(k + 1), _P(inv_T(odom[k]) @ odom[k + 1]), on))
    mn = gtsam.noiseModel.Diagonal.Sigmas(np.array([sig_marker[0]] * 3 + [sig_marker[1]] * 3))
    if robust:  # Huber: big (wrong) detections get down-weighted
        mn = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), mn)
    cands = {}
    for k, i, Tcm in obs:
        cands.setdefault(i, []).append(base[k] @ Tcm)
    tight = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3))
    for i, c in cands.items():
        if i in anchors:
            v.insert(_L(i), _P(anchors[i]))
            g.add(gtsam.PriorFactorPose3(_L(i), _P(anchors[i]), tight))
        else:
            v.insert(_L(i), _P(_init_marker(c)))
    for k, i, Tcm in obs:
        g.add(gtsam.BetweenFactorPose3(_X(k), _L(i), _P(Tcm), mn))
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(iters)
    res = gtsam.LevenbergMarquardtOptimizer(g, v, p).optimize()
    return dict(poses=[res.atPose3(_X(k)).matrix() for k in range(n)],
                markers={i: res.atPose3(_L(i)).matrix() for i in cands},
                err0=float(g.error(v)), err1=float(g.error(res)))


def relocalize(marker_map, dets):
    """Pose of the camera in the saved map frame from one frame's detections.
    T_W_C = T_W_M * inv(T_C_M), averaged over all known markers seen."""
    ests = [marker_map[i] @ inv_T(T) for i, T in dets if i in marker_map]
    if not ests:
        return None
    t = np.mean([e[:3, 3] for e in ests], axis=0)
    R = Rot.from_matrix(np.array([e[:3, :3] for e in ests])).mean().as_matrix()
    return make_T(R, t)
