import csv
import os
import sys
import time
import cv2

out = sys.argv[1] if len(sys.argv) > 1 else "data/run1"
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(0)
w, h = int(cap.get(3)), int(cap.get(4))
vw, rec, rows, k = None, False, [], 0
while True:
    ok, img = cap.read()
    if not ok:
        break
    t = time.monotonic()
    if rec:
        vw.write(img); rows.append((k, t)); k += 1
    show = img.copy()
    cv2.putText(show, "REC" if rec else "idle (r=start/stop q=quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if rec else (0, 255, 0), 2)
    cv2.imshow("record", show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        rec = not rec
        if rec and vw is None:
            vw = cv2.VideoWriter(f"{out}/video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    if key == ord("q"):
        break
if vw:
    vw.release()
with open(f"{out}/frame_times.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["frame", "t"]); wr.writerows(rows)
print(f"saved {k} frames to {out}")
