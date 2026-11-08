def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def associate_same_video(tid, plate_dets, frame_boxes, crowd_iou=0.2):
    """Same-video case: the plate was found inside this track's own box, so the link is direct.
    Confidence drops for frames where another motorcycle overlaps or contains the plate.
    frame_boxes: {frame: {track_id: bbox}}"""
    if not plate_dets:
        return {"confidence": None, "crowded_frames": 0,
                "note": "no rear-view plate observation for this track"}
    crowded = 0
    for d in plate_dets:
        px1, py1, px2, py2 = d["plate_bbox"]
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        for other, b in frame_boxes.get(d["frame"], {}).items():
            if other == tid:
                continue
            inside = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
            if inside or iou(d["moto_bbox"], b) > crowd_iou:
                crowded += 1
                break
    conf = round(0.95 * (1 - crowded / len(plate_dets)), 2)
    note = "direct same-video link"
    if crowded:
        note += f"; {crowded}/{len(plate_dets)} frames had an overlapping motorcycle"
    return {"confidence": conf, "crowded_frames": crowded, "note": note}


def classify(cross, phase, require_plate=False, plate_status="unreadable"):
    """Returns (status, reason). status: suspected_violation | not_violation | insufficient_evidence"""
    if cross["status"] == "uncertain":
        return "insufficient_evidence", f"crossing uncertain ({cross['reason']})"
    if phase["state"] == "UNKNOWN":
        return "insufficient_evidence", "no signal data covers the crossing time"
    if phase["near_transition"]:
        return "insufficient_evidence", f"crossing only {phase['margin_s']}s from a signal change"
    if phase["state"] == "GREEN":
        return "not_violation", "crossed on green"
    if phase["state"] == "YELLOW":
        return "not_violation", "crossed on yellow (not treated as a violation)"
    if require_plate and plate_status not in ("ok", "conflict"):
        return "insufficient_evidence", "red-light crossing but plate not identified"
    return "suspected_violation", "crossed on red"