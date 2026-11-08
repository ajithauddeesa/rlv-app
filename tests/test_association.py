from app.association import associate_same_video, classify

OK = {"status": "crossed", "reason": "ok", "time": 30.0, "confidence": 1.0}


def ph(state, near=False):
    return {"state": state, "margin_s": 1.0, "near_transition": near}


def test_green_not_violation():
    assert classify(OK, ph("GREEN"))[0] == "not_violation"


def test_yellow_not_violation():
    assert classify(OK, ph("YELLOW"))[0] == "not_violation"


def test_red_is_suspected_violation():
    assert classify(OK, ph("RED"))[0] == "suspected_violation"


def test_red_near_transition_insufficient():
    assert classify(OK, ph("RED", near=True))[0] == "insufficient_evidence"


def test_unknown_signal_insufficient():
    assert classify(OK, ph("UNKNOWN"))[0] == "insufficient_evidence"


def test_uncertain_crossing_insufficient():
    unc = {"status": "uncertain", "reason": "lost_near_line", "time": None, "confidence": 0.0}
    assert classify(unc, ph("RED"))[0] == "insufficient_evidence"


def test_red_with_unreadable_plate_follows_flag():
    assert classify(OK, ph("RED"), require_plate=False, plate_status="unreadable")[0] == "suspected_violation"
    assert classify(OK, ph("RED"), require_plate=True, plate_status="unreadable")[0] == "insufficient_evidence"


def test_association_penalises_overlapping_bike():
    dets = [{"frame": 5, "plate_bbox": [100, 100, 140, 120], "moto_bbox": [80, 50, 200, 200]}]
    alone = associate_same_video(1, dets, {5: {1: [80, 50, 200, 200]}})
    crowded = associate_same_video(1, dets, {5: {1: [80, 50, 200, 200], 2: [90, 60, 210, 210]}})
    assert alone["confidence"] > crowded["confidence"]


def test_association_no_plate():
    assert associate_same_video(1, [], {})["confidence"] is None