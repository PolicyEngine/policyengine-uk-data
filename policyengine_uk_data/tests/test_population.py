def test_child_limit():
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets import EnhancedFRS_2022_23

    # Initialize simulation
    sim = Microsimulation(dataset=EnhancedFRS_2022_23)

    population = sim.calculate("people", 2025).sum() / 1e6
    POPULATION_TARGET = 69.5  # Expected UK population in millions, per ONS 2022-based estimate here: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/bulletins/nationalpopulationprojections/2022based
    assert (
        abs(population / POPULATION_TARGET - 1) < 0.02
    ), f"Expected UK population of {POPULATION_TARGET:.1f} million, got {population:.1f} million."
