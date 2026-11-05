import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.simulate import simulate
from src.fusion import fuse, baseline_poses, relocalize
from src import evaluate as ev
from src.common import rot_err_deg


def analyse(sim, out="results", tag="nominal", robust=True, anchors=None, plot=True):
    gt, odom, obs = sim["gt"], sim["odom"], sim["obs"]
    base = baseline_poses(odom, gt[0])
    res = fuse(odom, obs, gt[0], anchors=anchors, robust=robust)
    mb, eb = ev.trajectory_metrics(base, gt)
    mf, ef = ev.trajectory_metrics(res["poses"], gt)
    lo, hi = sim["loss"]
    metrics = dict(
        tag=tag, baseline=mb, fused=mf,
        revisit_gap_baseline_m=ev.revisit_gap(base, gt), revisit_gap_fused_m=ev.revisit_gap(res["poses"], gt),
        marker_loss_window_mean_err_baseline_m=float(eb[lo:hi].mean()),
        marker_loss_window_mean_err_fused_m=float(ef[lo:hi].mean()),
        marker_map_error=ev.marker_errors(res["markers"], sim["markers_gt"]),
        n_observations=len(obs), keyframes_with_detection=len({k for k, _, _ in obs}) / len(gt),
        graph_error_before=res["err0"], graph_error_after=res["err1"])
    P = lambda ps: np.array([p[:3, 3] for p in ps])
    extra = dict(gt=P(gt), base=P(base), fused=P(res["poses"]), e_base=eb, e_fused=ef,
                 markers={i: T[:3, 3] for i, T in res["markers"].items()},
                 marker_T=res["markers"], metrics=metrics, loss=sim["loss"])
    if plot:
        os.makedirs(out, exist_ok=True)
        plot_run(extra, sim, f"{out}/{tag}.png")
    return metrics, extra


def plot_run(x, sim, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(*x["gt"][:, :2].T, "k-", lw=2, label="ground truth")
    ax[0].plot(*x["base"][:, :2].T, "r--", label="odometry only (baseline)")
    ax[0].plot(*x["fused"][:, :2].T, "g-", label="odometry + markers (factor graph)")
    gm = np.array([T[:3, 3] for T in sim["markers_gt"].values()])
    em = np.array(list(x["markers"].values()))
    ax[0].scatter(gm[:, 0], gm[:, 1], marker="s", c="k", s=60, label="marker (true)")
    ax[0].scatter(em[:, 0], em[:, 1], marker="x", c="b", s=60, label="marker (estimated)")
    ax[0].set_aspect("equal"); ax[0].set_title("Top view (room frame)"); ax[0].legend(fontsize=7)
    ax[1].plot(x["e_base"], "r--", label="baseline"); ax[1].plot(x["e_fused"], "g-", label="fused")
    ax[1].axvspan(*x["loss"], color="gray", alpha=0.25, label="no markers visible")
    ax[1].set_xlabel("keyframe"); ax[1].set_ylabel("position error e_p (m)")
    ax[1].set_title("Error vs ground truth"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def relocalization_test(marker_map, seed=7):
    sim = simulate(seed)  # a SECOND run in the same room, using the saved map
    by_k = {}
    for k, i, T in sim["obs"]:
        by_k.setdefault(k, []).append((i, T))
    k0 = min(by_k)
    T = relocalize(marker_map, by_k[k0])
    g = sim["gt"][k0]
    return dict(first_detection_keyframe=k0, time_to_relocalize_s=k0 * sim["dt"],
                position_error_m=float(np.linalg.norm(T[:3, 3] - g[:3, 3])),
                heading_error_deg=rot_err_deg(T[:3, :3], g[:3, :3]))


def experiments(out="results"):
    R = {}
    m, x = analyse(simulate(0), out, "exp1_3_nominal_loop_and_marker_loss")
    R["exp1_3_nominal"] = m
    R["exp4_relocalization"] = relocalization_test(x["marker_T"])
    R["exp5a_15pct_false_detections_robust"], _ = analyse(simulate(1, p_out=0.15), out, "exp5a_robust", True)
    R["exp5b_15pct_false_detections_NOT_robust"], _ = analyse(simulate(1, p_out=0.15), out, "exp5b_nonrobust", False)
    s = simulate(2)
    s["obs"] = [o for o in s["obs"] if o[1] in (0, 4)]
    R["exp5c_sparse_2_markers"], _ = analyse(s, out, "exp5c_sparse")
    json.dump(R, open(f"{out}/metrics.json", "w"), indent=2)
    return R


if __name__ == "__main__":
    R = experiments()
    print(f"{'experiment':42s} {'RMSE base':>10s} {'RMSE fused':>11s} {'end drift base':>15s} {'end drift fused':>16s}")
    for k, v in R.items():
        if "baseline" in v:
            print(f"{k:42s} {v['baseline']['rmse_m']:10.3f} {v['fused']['rmse_m']:11.3f} "
                  f"{v['baseline']['endpoint_drift_m']:15.3f} {v['fused']['endpoint_drift_m']:16.3f}")
    print("relocalization:", R["exp4_relocalization"])
    print("saved plots + metrics.json in ./results")
