def test_population(baseline):
    population = baseline.calculate("people", 2025).sum() / 1e6
    POPULATION_TARGET = 69.5  # ONS 2022-based projection for 2025, millions: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/bulletins/nationalpopulationprojections/2022based
    # Tightened from 7% to 4% after data-pipeline improvements in April 2026
    # (stage-2 QRF imputation #362, TFC target refresh #363, reported-anchor
    # takeup #359) pulled the weighted UK population down from ~74M (+6.5%)
    # to ~71M (+1.6% - 3.3% depending on stochastic calibration variance).
    # 4% headroom keeps CI stable across runs while still catching any
    # regression back toward the pre-April-2026 overshoot.
    assert abs(population / POPULATION_TARGET - 1) < 0.04, (
        f"Expected UK population of {POPULATION_TARGET:.1f} million, got {population:.1f} million."
    )
