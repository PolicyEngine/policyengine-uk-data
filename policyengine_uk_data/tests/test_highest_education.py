def test_highest_education(baseline):
    values = baseline.calculate("highest_education", period=2025)
    assert "Tertiary" in set(values)
    assert "Lower Secondary" in set(values)
    assert "Upper Secondary" in set(values)
