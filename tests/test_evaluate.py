from app.evaluate import score_plate


def test_exact():
    assert score_plate("AB123", "AB123")["outcome"] == "exact"


def test_abstained():
    assert score_plate("unreadable", "AB123")["outcome"] == "abstained"


def test_partial_all_correct():
    r = score_plate("?B1?3", "AB123")
    assert r["outcome"] == "partial_all_correct" and r["committed"] == 3


def test_wrong_char():
    assert score_plate("AX123", "AB123")["outcome"] == "has_wrong_chars"


def test_length_mismatch():
    assert score_plate("AB12", "AB123")["outcome"] == "length_mismatch"


def test_unknown_truth_chars_are_skipped():
    assert score_plate("AB123", "A?123")["committed"] == 4