import glob
import sys
import cv2
import numpy as np

folder, cols, rows, sq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
objs, imgs, size = [], [], None
for f in sorted(glob.glob(f"{folder}/*.*")):
    g = cv2.imread(f, 0)
    if g is None:
        continue
    size = g.shape[::-1]
    ok, c = cv2.findChessboardCorners(g, (cols, rows))
    if ok:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objs.append(objp); imgs.append(c)
print(f"used {len(objs)} images")
rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
print("reprojection rms (px):", rms, "\nK=\n", K, "\ndist=", dist.ravel())
np.savez("calib.npz", K=K, dist=dist, rms=rms, size=size)
