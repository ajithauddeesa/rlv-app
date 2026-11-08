import json
import os
import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO

MARGIN = 0.10      # extra margin around the motorcycle box
STEP = 2           # process every Nth frame
PLATE_CONF = 0.15  # low on purpose, we filter later


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def detect_plates(video, tracks_json, out_json,
                  model_path="models/plate_detector.pt",
                  crop_dir="outputs/crops", frame_dir="outputs/frames/evidence"):
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    model = YOLO(model_path)
    print("plate model classes:", model.names)

    data = json.load(open(tracks_json))
    by_frame = defaultdict(list)
    for tid, pts in data["tracks"].items():
        for p in pts:
            if p["frame"] % STEP == 0:
                by_frame[p["frame"]].append((int(tid), p))

    cap = cv2.VideoCapture(video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result = defaultdict(list)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        items = by_frame.get(idx)
        if items:
            annotated = None
            for tid, p in items:
                x1, y1, x2, y2 = p["bbox"]
                mw, mh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
                cx1, cy1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
                cx2, cy2 = min(W, int(x2 + mw)), min(H, int(y2 + mh))
                if cx2 - cx1 < 20 or cy2 - cy1 < 20:
                    continue
                roi = frame[cy1:cy2, cx1:cx2]
                r = model.predict(roi, conf=PLATE_CONF, verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                # keep the most confident plate, prefer lower half of the bike
                mcx = (x1 + x2) / 2 - cx1          # motorcycle centre x, in crop coordinates
                mw_box = max(1.0, x2 - x1)

                def score(b):
                    bx = (float(b.xyxy[0][0]) + float(b.xyxy[0][2])) / 2
                    return float(b.conf) - 0.5 * abs(bx - mcx) / mw_box

                best = max(r.boxes, key=score)
                bx1, by1, bx2, by2 = [float(v) for v in best.xyxy[0]]
                fx1, fy1, fx2, fy2 = int(bx1 + cx1), int(by1 + cy1), int(bx2 + cx1), int(by2 + cy1)
                pad = 4
                crop = frame[max(0, fy1 - pad):min(H, fy2 + pad), max(0, fx1 - pad):min(W, fx2 + pad)]
                if crop.size == 0:
                    continue
                name = f"track{tid}_f{idx}.jpg"
                cv2.imwrite(os.path.join(crop_dir, name), crop)
                result[tid].append({
                    "frame": idx, "t": p["t"], "plate_bbox": [fx1, fy1, fx2, fy2],
                    "det_conf": round(float(best.conf), 3),
                    "sharpness": round(sharpness(crop), 1),
                    "crop": f"{crop_dir}/{name}",
                    "moto_bbox": [round(v, 1) for v in p["bbox"]],
                })
                if annotated is None:
                    annotated = frame.copy()
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
                cv2.putText(annotated, f"id{tid}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if annotated is not None:
                cv2.imwrite(os.path.join(frame_dir, f"frame_{idx:05d}.jpg"), annotated)
        idx += 1
    cap.release()

    json.dump({"plates": result}, open(out_json, "w"))
    print(f"\n{len(result)} of {len(data['tracks'])} tracks have at least one plate detection\n")
    print(f"{'track':>5} {'sampled':>8} {'with_plate':>10} {'best_conf':>9} {'max_crop_w':>10} {'best_sharp':>10}")
    for tid in sorted(data["tracks"], key=int):
        n_sampled = sum(1 for p in data["tracks"][tid] if p["frame"] % STEP == 0)
        dets = result.get(int(tid), [])
        if dets:
            wmax = max(d["plate_bbox"][2] - d["plate_bbox"][0] for d in dets)
            print(f"{tid:>5} {n_sampled:>8} {len(dets):>10} {max(d['det_conf'] for d in dets):>9} "
                  f"{wmax:>10} {max(d['sharpness'] for d in dets):>10}")
        else:
            print(f"{tid:>5} {n_sampled:>8} {0:>10} {'-':>9} {'-':>10} {'-':>10}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/clip01_junction.mp4"
    detect_plates(video, "outputs/tracks_clip01.json", "outputs/plates_clip01.json")