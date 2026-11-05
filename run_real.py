import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.fusion import fuse, baseline_poses, relocalize
from src.vo_simple import run_vo
from src.imu_dr import dead_reckon
from src import evaluate as ev
from src.common import inv_T, make_T


def run(d, step=3, map_path=None, closed=False, out="results_real"):
    os.makedirs(out, exist_ok=True)
    c = np.load(f"{d}/calib.npz")
    K, dist = c["K"], c["dist"]
    by_frame = {}
    for r in json.load(open(f"{d}/markers.json")):
        by_frame.setdefault(r["frame"], {})[r["id"]] = np.array(r["T"])
    imu_csv = f"{d}/imu.csv" if os.path.exists(f"{d}/imu.csv") else None
    if os.path.exists(f"{d}/odom.npz"):
        z = np.load(f"{d}/odom.npz")
        idxs, odom = list(z["idxs"]), list(z["poses"])
    else:
        idxs, odom = run_vo(f"{d}/video.mp4", K, dist, by_frame, step=step, imu_csv=imu_csv)
        np.savez(f"{d}/odom.npz", idxs=np.array(idxs), poses=np.array(odom))
    obs = [(k, i, T) for k, f in enumerate(idxs) for i, T in by_frame.get(int(f), {}).items()]

    # Align OpenCV camera frame (x-right, y-down, z-forward) with Room frame (X-right, Y-forward, Z-up)
    R_W_C0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    first, anchors, reloc = make_T(R_W_C0, [0.0, 0.0, 1.4]), None, None

    if map_path and os.path.exists(map_path):
        mp = {int(k): np.array(v) for k, v in json.load(open(map_path)).items()}
        for k in range(len(idxs)):
            P = relocalize(mp, [(i, T) for kk, i, T in obs if kk == k])
            if P is not None:
                first = P @ inv_T(inv_T(odom[0]) @ odom[k])
                reloc, anchors = dict(keyframe=k, frame=int(idxs[k])), mp
                break

    base = baseline_poses(odom, first)
    res = fuse(odom, obs, first, anchors=anchors, prior_sigma=0.05 if anchors else 1e-3)
    P = lambda ps: np.array([p[:3, 3] for p in ps])
    Pb, Pf = P(base), P(res["poses"])

    m = dict(
        keyframes=len(idxs),
        observations=len(obs),
        unique_markers_mapped=len(res["markers"]),
        keyframes_with_detection=len({k for k, _, _ in obs}) / max(len(idxs), 1),
        path_length_estimated_m=float(np.sum(np.linalg.norm(np.diff(Pf, axis=0), axis=1))),
        graph_error_before=res["err0"],
        graph_error_after=res["err1"],
        imu_stream_used=bool(imu_csv),
        relocalization=reloc,
    )
    if imu_csv:
        try:
            _, P_imu = dead_reckon(imu_csv)
            m["imu_dead_reckoning_endpoint_drift_m"] = float(np.linalg.norm(P_imu[-1] - P_imu[0]))
        except Exception:
            pass
    if closed:
        m["loop_gap_baseline_m"] = float(np.linalg.norm(Pb[-1] - Pb[0]))
        m["loop_gap_fused_m"] = float(np.linalg.norm(Pf[-1] - Pf[0]))
    if os.path.exists(f"{d}/tape.json"):
        m["marker_pair_error"] = ev.marker_pair_errors(res["markers"], json.load(open(f"{d}/tape.json")))

    json.dump({int(i): T.tolist() for i, T in res["markers"].items()}, open(f"{out}/marker_map.json", "w"), indent=2)
    np.savetxt(f"{out}/trajectory_fused.csv", Pf, delimiter=",", header="x,y,z")
    json.dump(m, open(f"{out}/metrics.json", "w"), indent=2, default=float)

    mk = np.array([T[:3, 3] for T in res["markers"].values()]) if res["markers"] else np.empty((0, 3))
    plt.figure(figsize=(7, 6))
    plt.plot(*Pb[:, :2].T, "r--", lw=1.8, label="odometry only (baseline)")
    plt.plot(*Pf[:, :2].T, "g-", lw=2.2, label="with ArUco markers (factor graph)")
    if len(mk) > 0:
        plt.scatter(mk[:, 0], mk[:, 1], marker="s", s=70, c="blue", label="estimated markers")
        for i, T in res["markers"].items():
            plt.annotate(f"M{i}", (T[0, 3], T[1, 3]), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xlabel("Room X (m)")
    plt.ylabel("Room Y (m)")
    plt.title("Real Dataset Trajectory & Marker Map (Top View)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/trajectory.png", dpi=120)
    plt.close()
    return dict(base=Pb, fused=Pf, markers={i: T[:3, 3] for i, T in res["markers"].items()}, metrics=m)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("dataset")
    a.add_argument("--step", type=int, default=3)
    a.add_argument("--map")
    a.add_argument("--closed", action="store_true")
    a = a.parse_args()
    print(json.dumps(run(a.dataset, a.step, a.map, a.closed)["metrics"], indent=2, default=float))