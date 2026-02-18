def test_highest_education(baseline):
    values = baseline.calculate("highest_education", period=2025)
    assert len(values) > 0
