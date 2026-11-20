# Red-Light Event Analysis & Rear-View Plate Recognition

Detects suspected motorcycle red-light violations in CCTV video and reads the rear licence
plate: motorcycle tracking, stop-line crossing, signal comparison, automatic rear-plate
localisation, multi-frame OCR, and plate-to-event association, behind a web UI.

## Quick start (Docker, CPU)

    docker compose up --build

Open http://localhost:8080, click **Use sample clip**, then **Run analysis**.
The first build takes several minutes (PyTorch and model weights are downloaded at build time,
so the container then runs offline). Make sure nothing else is using port 8080.

## Run locally (Python 3.11)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    pytest -q
    uvicorn app.main:app --port 8080

## Optional GPU

Install the CUDA build of PyTorch in the Dockerfile instead of the CPU wheel, install the
NVIDIA Container Toolkit on the host, and uncomment the `deploy` block in `compose.yaml`.
Not tested here.

## Pipeline

1. **Tracking** (`app/detection.py`): YOLOv8n + ByteTrack, COCO class "motorcycle".
2. **Stop-line crossing** (`app/stopline.py`): signed distance of each bike's bottom-centre to a
   configurable line, a dead band, a minimum number of frames on each side, and an
   interpolated crossing time. Outcomes: crossed / not_crossed / uncertain (lost near the
   line, jitter, gap, too few frames).
3. **Signal** (`app/signal_phase.py`): phase lookup with a clock offset. Crossings within
   +-0.25 s of a phase change, or outside the signal data, are flagged.
4. **Plate localisation** (`app/plates.py`): a pretrained YOLO plate detector
   (`models/plate_detector.pt`, from Hugging Face `yasirfaizahmed/license-plate-object-detection`)
   runs inside each motorcycle box on every 2nd frame.
5. **OCR** (`app/ocr.py`): `fast-plate-ocr` (`cct-s-v2-global-model`) on each crop, then
   per-character confidence-weighted voting across frames. Disagreeing positions become `?`,
   too little evidence becomes `unreadable`. Characters are never guessed.
6. **Association and classification** (`app/association.py`, `app/pipeline.py`): the plate
   found inside a track's box is linked to that track; association confidence drops when
   another motorcycle overlaps the plate. Verdicts: `suspected_violation`, `not_violation`,
   `insufficient_evidence`, each with an explanation.
7. **Export**: JSON and CSV (UI buttons, or `/export.json`, `/export.csv`).

## Using the UI

Load a video (or the sample), click two points to draw the stop line, check or edit the
signal phases and clock offset, run the analysis, click an event to see the original
rear-view frame beside the plate crop and the per-frame OCR reads.

## Design decisions

- One uploaded video serves as both the traffic view and the rear-view OCR footage (allowed
  by the brief), so association is a direct track link. A separate-camera matching step
  (time window plus a lane/position score, ambiguity when the top two scores are close)
  is NOT implemented.
- Yellow is not treated as a violation.
- An unreadable plate does not hide a red-light crossing: the event is still reported as
  `suspected_violation` with plate `unreadable`. Set `REQUIRE_PLATE_FOR_VIOLATION = True`
  in `app/pipeline.py` to make it `insufficient_evidence` instead.
- The stop line is a virtual line drawn at the bottom of the frame, where bikes are largest
  and plates most legible, not the painted line.
- Signal phases were read by hand from the lights visible in the clip
  (`data/signals/clip01_signal.json`).

## Evaluation (sample clip, 35 s, 1080p, 30 fps)

TODO: paste the output of `python -m app.evaluate` and the speed line from the UI.
The sample is tiny (about 15 crossings, 5 tracks with plates) and almost all crossings are
on green, so these numbers say little about general performance.

## Known limitations

- Plates 40 to 70 px wide are too blurry for reliable OCR: most plates come out
  `unreadable` or partly read (for example `?G?0991`).
- The plate detector was trained mostly on cars and misses some motorcycle plates, including
  the one on the red-light track (22). Plates hidden by top boxes or riders are not found
  (correctly).
- Static camera assumed; a moving camera breaks the fixed stop line.
- Only one red-light crossing in the sample clip, so violation detection is barely tested.
- Tracking can fragment or lose small distant bikes (yolov8n).
- Signal times are manual, not read from the video.
- Separate-camera association is not implemented.
- GPU path is untested.