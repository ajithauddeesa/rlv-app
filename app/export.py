import csv
import json

COLUMNS = ["event_id", "track_id", "status", "explanation", "crossing_time_s", "signal_state",
           "signal_time_s", "margin_to_change_s", "near_transition", "crossing_status",
           "crossing_reason", "crossing_confidence", "plate_text", "plate_status",
           "ocr_confidence", "ocr_frames_used", "association_confidence", "association_note",
           "rear_frames", "rear_frame_images", "plate_crops"]


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, events):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for e in events:
            row = dict(e)
            row["rear_frames"] = ";".join(str(x["frame"]) for x in e["rear_evidence"])
            row["rear_frame_images"] = ";".join(x["frame_image"] for x in e["rear_evidence"])
            row["plate_crops"] = ";".join(x["crop"] for x in e["rear_evidence"])
            w.writerow(row)