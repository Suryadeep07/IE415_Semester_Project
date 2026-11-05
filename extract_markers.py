import json
import sys
import cv2
import numpy as np
from src.aruco_utils import make_detector, detect_poses


def main(d, step=3):
    meta = json.load(open(f"{d}/meta.json"))
    c = np.load(f"{d}/calib.npz")
    K, dist = c["K"], c["dist"]
    size = meta["marker_size_m"]
    det = make_detector(meta.get("dictionary", "DICT_4X4_50"))
    cap = cv2.VideoCapture(f"{d}/video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(f"{d}/annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))
    recs, f, n_proc, n_det, saved = [], -1, 0, 0, False
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if f % step:
            continue
        n_proc += 1
        dets = detect_poses(img, det, K, dist, size)
        n_det += bool(dets)
        for dd in dets:
            recs.append(dict(frame=f, t=f / fps, id=dd["id"], T=dd["T"].tolist(), err=dd["err"]))
            cv2.drawFrameAxes(img, K, dist, dd["rvec"], dd["tvec"], size / 2)
            cv2.putText(img, f"id {dd['id']} d={np.linalg.norm(dd['tvec']):.2f}m", (10, 30 + 25 * dd["id"] % 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        vw.write(img)
        if dets and not saved:
            cv2.imwrite(f"{d}/preview.jpg", img); saved = True
    vw.release()
    json.dump(recs, open(f"{d}/markers.json", "w"))
    stats = dict(frames_processed=n_proc, frames_with_marker=n_det, detection_rate=n_det / max(n_proc, 1),
                 detections=len(recs))
    print(stats)
    return stats


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
