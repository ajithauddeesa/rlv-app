import json
from collections import defaultdict

from app.association import associate_same_video, classify
from app.export import write_csv, write_json
from app.signal_phase import Signal
from app.stopline import StopLine, detect_crossing

REQUIRE_PLATE_FOR_VIOLATION = False


def build_events(tracks_data, plates, ocr, line, signal, require_plate=REQUIRE_PLATE_FOR_VIOLATION):
    tracks = tracks_data["tracks"]
    frame_boxes = defaultdict(dict)
    for tid, pts in tracks.items():
        for p in pts:
            frame_boxes[p["frame"]][int(tid)] = p["bbox"]

    events, ignored = [], []
    for tid in sorted(tracks, key=int):
        pts = tracks[tid]
        points = [(p["t"], (p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]) for p in pts]
        cross = detect_crossing(points, line)
        if cross["status"] == "not_crossed":
            ignored.append({"track_id": int(tid), "reason": cross["reason"]})
            continue

        if cross["status"] == "crossed":
            phase = signal.lookup(cross["time"])
        else:
            phase = {"state": "UNKNOWN", "signal_time": None, "margin_s": None, "near_transition": False}

        o = ocr.get(tid)
        plate_text = o["text"] if o else "unreadable"
        plate_status = o["status"] if o else "no_plate_detected"
        dets = plates.get(tid, [])
        assoc = associate_same_video(int(tid), dets, frame_boxes)
        status, reason = classify(cross, phase, require_plate, plate_status)

        crossing_txt = (f"crossed the stop line at {cross['time']:.2f}s "
                        f"(confidence {cross['confidence']})" if cross["status"] == "crossed"
                        else f"crossing could not be confirmed ({cross['reason']})")
        if plate_status == "ok":
            plate_txt = f"plate read as {plate_text} from {o['n_used']} frames"
        elif plate_status == "conflict":
            plate_txt = f"plate partly read as {plate_text} ('?' = frames disagree), not reliable"
        elif plate_status == "unreadable":
            plate_txt = "plate seen but unreadable"
        else:
            plate_txt = "no plate could be localised on this motorcycle"
        margin_txt = "" if phase["margin_s"] is None else " (margin " + str(phase["margin_s"]) + "s)"
        explanation = (f"Track {tid} {crossing_txt}. Signal: {phase['state']}{margin_txt}. "
                       f"Verdict: {reason}. {plate_txt.capitalize()}.")
        events.append({
            "event_id": f"E{len(events) + 1:03d}",
            "track_id": int(tid),
            "status": status,
            "explanation": explanation,
            "crossing_time_s": cross["time"],
            "signal_state": phase["state"],
            "signal_time_s": phase["signal_time"],
            "margin_to_change_s": phase["margin_s"],
            "near_transition": phase["near_transition"],
            "crossing_status": cross["status"],
            "crossing_reason": cross["reason"],
            "crossing_confidence": cross["confidence"],
            "plate_text": plate_text,
            "plate_status": plate_status,
            "ocr_confidence": o["confidence"] if o else 0.0,
            "ocr_frames_used": o["n_used"] if o else 0,
            "association_confidence": assoc["confidence"],
            "association_note": assoc["note"],
            "rear_evidence": [{"frame": d["frame"], "t": d["t"], "plate_bbox": d["plate_bbox"],
                               "det_conf": d["det_conf"], "crop": d["crop"],
                               "frame_image": f"outputs/frames/evidence/frame_{d['frame']:05d}.jpg"}
                              for d in dets],
        })
    return events, ignored


def run(tracks="outputs/tracks_clip01.json", plates="outputs/plates_clip01.json",
        ocr="outputs/ocr_clip01.json", signal="data/signals/clip01_signal.json",
        line=(0, 1000, 1230, 1000), out_prefix="outputs/events_clip01"):
    tracks_data = json.load(open(tracks))
    plates_d = json.load(open(plates))["plates"]
    ocr_d = json.load(open(ocr))
    sig = Signal.from_json(signal)
    events, ignored = build_events(tracks_data, plates_d, ocr_d, StopLine(*line, approach_sign=1), sig)
    write_json(out_prefix + ".json", {
        "video_fps": tracks_data["fps"], "signal_approach": sig.approach,
        "clock_offset_s": sig.clock_offset_s, "events": events, "ignored_tracks": ignored})
    write_csv(out_prefix + ".csv", events)
    for e in events:
        print(f"{e['event_id']} track {e['track_id']:>3} t={e['crossing_time_s']} {e['signal_state']:<7} "
              f"{e['status']:<20} plate={e['plate_text']:<10} assoc={e['association_confidence']}")
    print(f"\n{len(events)} events, {len(ignored)} ignored tracks -> {out_prefix}.json / .csv")


if __name__ == "__main__":
    run()