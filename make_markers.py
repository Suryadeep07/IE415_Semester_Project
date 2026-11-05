import sys
import cv2
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for i in range(n):
    img = cv2.aruco.generateImageMarker(d, i, 600)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)  # white margin is required
    cv2.putText(img, f"ID {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.imwrite(f"marker_{i}.png", img)
print("wrote marker_*.png")
