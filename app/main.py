import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)  # the pipeline modules use project-relative paths

from app import ocr as ocr_mod          # noqa: E402
from app import pipeline                # noqa: E402
from app.detection import track_video   # noqa: E402
from app.plates import detect_plates    # noqa: E402

UPLOAD = Path("data/uploads")
UPLOAD.mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

VIDEO = UPLOAD / "current.mp4"
SAMPLE = Path("data/videos/sample_short.mp4")
SIGNAL_ACTIVE = Path("data/signals/active.json")
SIGNAL_DEFAULT = Path("data/signals/clip01_signal.json")
LINE_FILE = Path("data/stopline.json")
DEFAULT_LINE = [0.0, 1000.0, 1230.0, 1000.0]
OUT_PREFIX = "outputs/events"
TRACKS, PLATES, OCR = "outputs/tracks.json", "outputs/plates.json", "outputs/ocr.json"

STATE = {"state": "idle", "stage": "", "error": None}
LOCK = threading.Lock()

app = FastAPI(title="Red-light event analysis")


class LineIn(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


def video_info():
    if not VIDEO.exists():
        return None
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "frames": n,
            "duration_s": round(n / fps, 2) if fps else 0}
    cap.release()
    return info


def load_line():
    if LINE_FILE.exists():
        return json.load(open(LINE_FILE))
    return DEFAULT_LINE


def load_signal():
    p = SIGNAL_ACTIVE if SIGNAL_ACTIVE.exists() else SIGNAL_DEFAULT
    if p.exists():
        return json.load(open(p))
    return {"approach": "main", "clock_offset_s": 0.0, "transition_uncertainty_s": 0.25, "phases": []}


def clear_results():
    for ext in (".json", ".csv"):
        Path(OUT_PREFIX + ext).unlink(missing_ok=True)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/video/info")
def get_video_info():
    info = video_info()
    return {"loaded": False} if not info else {"loaded": True, **info}


@app.post("/api/video")
def upload_video(file: UploadFile = File(...)):
    if STATE["state"] == "running":
        raise HTTPException(409, "analysis is running")
    with open(VIDEO, "wb") as out:
        shutil.copyfileobj(file.file, out)
    clear_results()
    info = video_info()
    if not info or not info["frames"]:
        VIDEO.unlink(missing_ok=True)
        raise HTTPException(400, "could not read that video file")
    return {"loaded": True, **info}


@app.post("/api/sample")
def use_sample():
    if STATE["state"] == "running":
        raise HTTPException(409, "analysis is running")
    if not SAMPLE.exists():
        raise HTTPException(404, "sample clip not found at data/videos/sample_short.mp4")
    shutil.copyfile(SAMPLE, VIDEO)
    clear_results()
    return {"loaded": True, **video_info()}


@app.get("/api/video")
def stream_video(request: Request):
    if not VIDEO.exists():
        raise HTTPException(404, "no video")
    size = VIDEO.stat().st_size
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(VIDEO, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    m = re.match(r"bytes=(\d*)-(\d*)", rng)
    start = int(m.group(1)) if m and m.group(1) else 0
    end = int(m.group(2)) if m and m.group(2) else size - 1
    end = min(end, size - 1)
    if start > end:
        raise HTTPException(416, "bad range")
    length = end - start + 1

    def chunks():
        with open(VIDEO, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                data = f.read(min(1 << 20, left))
                if not data:
                    break
                left -= len(data)
                yield data

    return StreamingResponse(chunks(), status_code=206, media_type="video/mp4",
                             headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                      "Accept-Ranges": "bytes", "Content-Length": str(length)})


@app.get("/api/frame")
def get_frame(t: float = 0.0):
    if not VIDEO.exists():
        raise HTTPException(404, "no video")
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, t) * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "frame not available")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/stopline")
def get_stopline():
    x1, y1, x2, y2 = load_line()
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@app.post("/api/stopline")
def set_stopline(body: LineIn):
    x1, y1, x2, y2 = body.x1, body.y1, body.x2, body.y2
    if (x1, y1) == (x2, y2):
        raise HTTPException(400, "the stop line needs two different points")
    if x1 > x2:  # keep left-to-right so the 'approach' side is below the line
        x1, y1, x2, y2 = x2, y2, x1, y1
    LINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump([x1, y1, x2, y2], open(LINE_FILE, "w"))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@app.get("/api/signal")
def get_signal():
    return load_signal()


@app.post("/api/signal")
def set_signal(body: dict):
    phases = body.get("phases")
    if not isinstance(phases, list) or not phases:
        raise HTTPException(400, "'phases' must be a non-empty list")
    clean = []
    for p in phases:
        try:
            start, end, state = float(p["start"]), float(p["end"]), str(p["state"]).upper()
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "each phase needs numeric start/end and a state")
        if end <= start or state not in ("GREEN", "YELLOW", "RED"):
            raise HTTPException(400, f"invalid phase: {p}")
        clean.append({"start": start, "end": end, "state": state})
    try:
        data = {"approach": str(body.get("approach", "main")),
                "clock_offset_s": float(body.get("clock_offset_s", 0.0)),
                "transition_uncertainty_s": float(body.get("transition_uncertainty_s", 0.25)),
                "phases": clean}
    except (TypeError, ValueError):
        raise HTTPException(400, "clock_offset_s and transition_uncertainty_s must be numbers")
    SIGNAL_ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(SIGNAL_ACTIVE, "w"), indent=2)
    return data


def _run(line, signal_path, info):
    stages = {}
    t_all = time.time()

    def stage(name):
        STATE["stage"] = name
        return time.time()

    try:
        t = stage("tracking motorcycles")
        track_video(str(VIDEO), TRACKS)
        stages["tracking"] = round(time.time() - t, 1)

        t = stage("localising rear plates")
        for d in ("outputs/crops", "outputs/frames/evidence"):
            shutil.rmtree(d, ignore_errors=True)
        detect_plates(str(VIDEO), TRACKS, PLATES)
        stages["plate_detection"] = round(time.time() - t, 1)

        t = stage("reading plates")
        ocr_mod.run(PLATES, OCR)
        stages["ocr"] = round(time.time() - t, 1)

        t = stage("classifying events")
        pipeline.run(tracks=TRACKS, plates=PLATES, ocr=OCR, signal=str(signal_path),
                     line=tuple(line), out_prefix=OUT_PREFIX)
        stages["association"] = round(time.time() - t, 1)

        total = time.time() - t_all
        path = Path(OUT_PREFIX + ".json")
        data = json.load(open(path))
        data["run_info"] = {
            "video_duration_s": info["duration_s"], "frames": info["frames"],
            "processing_s": round(total, 1),
            "processing_fps": round(info["frames"] / total, 2) if total else None,
            "seconds_per_video_second": round(total / info["duration_s"], 2) if info["duration_s"] else None,
            "stage_seconds": stages, "stop_line": list(line), "device": "cpu"}
        json.dump(data, open(path, "w"), indent=2)
        STATE.update(state="done", stage="")
    except Exception as e:  # report to the UI instead of dying silently
        STATE.update(state="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/analyze")
def analyze():
    info = video_info()
    if not info:
        raise HTTPException(400, "upload a video (or use the sample clip) first")
    with LOCK:
        if STATE["state"] == "running":
            raise HTTPException(409, "analysis already running")
        STATE.update(state="running", stage="starting", error=None)
    sig_path = SIGNAL_ACTIVE if SIGNAL_ACTIVE.exists() else SIGNAL_DEFAULT
    threading.Thread(target=_run, args=(load_line(), sig_path, info), daemon=True).start()
    return {"started": True}


@app.get("/api/status")
def status():
    return STATE


@app.get("/api/events")
def events():
    p = Path(OUT_PREFIX + ".json")
    if not p.exists():
        return {"events": [], "ignored_tracks": []}
    return json.load(open(p))


@app.get("/export.json")
def export_json():
    p = Path(OUT_PREFIX + ".json")
    if not p.exists():
        raise HTTPException(404, "run an analysis first")
    return FileResponse(p, media_type="application/json", filename="red_light_events.json")


@app.get("/export.csv")
def export_csv():
    p = Path(OUT_PREFIX + ".csv")
    if not p.exists():
        raise HTTPException(404, "run an analysis first")
    return FileResponse(p, media_type="text/csv", filename="red_light_events.csv")


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")