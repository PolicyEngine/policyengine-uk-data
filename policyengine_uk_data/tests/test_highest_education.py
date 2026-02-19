from policyengine_uk import Microsimulation


def test_highest_education(frs):
    sim = Microsimulation(dataset=frs)
    values = sim.calculate("highest_education", period=2025)
    assert "TERTIARY" in set(values)
    assert "LOWER_SECONDARY" in set(values)
    assert "UPPER_SECONDARY" in set(values)


def test_highest_education_enhanced_frs(enhanced_frs):
    sim = Microsimulation(dataset=enhanced_frs)
    values = sim.calculate("highest_education", period=2025)
    assert "TERTIARY" in set(values)
    assert "LOWER_SECONDARY" in set(values)
    assert "UPPER_SECONDARY" in set(values)
