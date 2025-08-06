from policyengine_uk.system import parameters


def get_population_growth_factor(start_year: int, end_year: int) -> float:
    """
    Calculate the population growth factor between two years.

    Args:
        start_year (int): The starting year.
        end_year (int): The ending year.

    Returns:
        float: The population growth factor.
    """

    population = parameters.gov.economic_assumptions.indices.ons.population

    return population(end_year) / population(start_year)
