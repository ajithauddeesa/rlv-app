import json
import sys
import cv2
from ultralytics import YOLO

MOTORCYCLE = 3  # COCO class id


def track_video(video_path, out_json, model_name="yolov8n.pt", conf=0.25):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    tracks = {}
    results = model.track(
        source=video_path,
        classes=[MOTORCYCLE],
        tracker="bytetrack.yaml",
        conf=conf,
        persist=True,
        stream=True,
        verbose=False,
    )
    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            continue
        for box, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.conf.tolist()):
            tracks.setdefault(int(tid), []).append({
                "frame": frame_idx,
                "t": round(frame_idx / fps, 3),
                "bbox": [round(v, 1) for v in box],
                "conf": round(c, 3),
            })

    with open(out_json, "w") as f:
        json.dump({"fps": fps, "tracks": tracks}, f)
    print(f"{len(tracks)} tracks saved to {out_json}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/clip01_junction.mp4"
    track_video(video, "outputs/tracks_clip01.json")