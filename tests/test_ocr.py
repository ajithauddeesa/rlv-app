from app.ocr import fuse_readings


def reads(*items, conf=0.9):
    return [{"text": t, "conf": conf} for t in items]


def test_agreement():
    r = fuse_readings(reads("AB123", "AB123", "AB123"))
    assert (r["text"], r["status"]) == ("AB123", "ok")


def test_single_noisy_read_is_outvoted():
    r = fuse_readings(reads("AB123", "AB123", "AB123", "AB123", "AX123"))
    assert (r["text"], r["status"]) == ("AB123", "ok")


def test_conflict_marks_question_mark_not_guess():
    r = fuse_readings(reads("AB123", "AB123", "AX123", "AX123"))
    assert r["text"] == "A?123" and r["status"] == "conflict"


def test_too_few_reads_unreadable():
    assert fuse_readings(reads("AB123", "AB123"))["text"] == "unreadable"


def test_low_confidence_reads_ignored():
    r = fuse_readings(reads("AB123", "AB123") + reads("ZZZZZ", "ZZZZZ", "ZZZZZ", conf=0.1))
    assert r["status"] == "unreadable"


def test_most_supported_length_wins():
    r = fuse_readings(reads("AB123", "AB123", "AB123", "AB1234"))
    assert r["text"] == "AB123"