import json
import cv2

VIDEO = "data/videos/clip01_junction.mp4"
TRACKS = "outputs/tracks_clip01.json"

# Candidate stop line (x1, y1, x2, y2) in 1920x1080 pixels. Adjust after viewing.
LINE = (0, 1000, 1230, 1000)

data = json.load(open(TRACKS))
cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()

# Draw every track's bottom-centre path on the first frame
for tid, pts in data["tracks"].items():
    path = [(int((p["bbox"][0] + p["bbox"][2]) / 2), int(p["bbox"][3])) for p in pts]
    for a, b in zip(path, path[1:]):
        cv2.line(frame, a, b, (0, 255, 255), 2)
    cv2.putText(frame, tid, path[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.line(frame, LINE[:2], LINE[2:], (0, 0, 255), 4)
cv2.imwrite("outputs/frames/line_check.jpg", frame)
print("saved outputs/frames/line_check.jpg")