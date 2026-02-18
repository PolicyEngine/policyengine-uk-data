from policyengine_uk_data.storage import STORAGE_FOLDER
import pandas as pd
from policyengine_uk.data import UKSingleYearDataset

START_YEAR = 2020
END_YEAR = 2034


def create_policyengine_uprating_factors_table():
    from policyengine_uk.system import system

    df = pd.DataFrame()

    variable_names = []
    years = []
    index_values = []

    for variable in system.variables.values():
        if variable.uprating is not None:
            parameter = system.parameters.get_child(variable.uprating)
            start_value = parameter(START_YEAR)
            for year in range(START_YEAR, END_YEAR + 1):
                variable_names.append(variable.name)
                years.append(year)
                growth = parameter(year) / start_value
                index_values.append(round(growth, 3))

    df["Variable"] = variable_names
    df["Year"] = years
    df["Value"] = index_values

    # Convert to there is a column for each year
    df = df.pivot(index="Variable", columns="Year", values="Value")
    df = df.sort_values("Variable")
    df.to_csv(STORAGE_FOLDER / "uprating_factors.csv")

    # Create a table with growth factors by year

    df_growth = df.copy()
    for year in range(END_YEAR, START_YEAR, -1):
        df_growth[year] = round(df_growth[year] / df_growth[year - 1] - 1, 3)
    df_growth[START_YEAR] = 0

    df_growth.to_csv(STORAGE_FOLDER / "uprating_growth_factors.csv")
    return df


def uprate_values(values, variable_name, start_year=2020, end_year=2034):
    uprating_factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    uprating_factors = uprating_factors.set_index("Variable")
    uprating_factors = uprating_factors.loc[variable_name]

    initial_index = uprating_factors[str(start_year)]
    end_index = uprating_factors[str(end_year)]
    relative_change = end_index / initial_index

    return values * relative_change


# FRS-weighted GB targets for total Plan 2 and Plan 5 outstanding borrowers
# (including below-threshold). Derived from DfE student loan forecasts (England),
# scaled to GB (÷0.84) and adjusted for FRS coverage (55.9% of total outstanding
# borrowers captured, calibrated from 2023 base).
# Plan 2 closed to new entrants after Sept 2023; growth reflects new cohorts
# becoming graduates and entering outstanding-loan status. Plan 5 is the new
# post-2023 cohort. Figures from 2030 are extrapolated from DfE trend.
_PLAN_TARGETS = {
    # year: (plan_2_millions, plan_5_millions)
    2024: (5.950, 0.007),
    2025: (6.462, 0.153),
    2026: (6.895, 0.419),
    2027: (7.065, 0.918),
    2028: (7.055, 1.571),
    2029: (7.005, 2.263),
    2030: (6.955, 2.995),
    2031: (6.855, 3.627),
    2032: (6.705, 4.160),
    2033: (6.506, 4.592),
    2034: (6.256, 4.992),
}

# Plan 1 write-off cutoff by year: loan term is 25 years post-graduation.
# Assuming graduation ~age 21, write-off at age 21+25+3=49... but the standard
# rule is 25 years from the April after graduation. For a person who started
# in 1998 (age 18) and graduated 2001, write-off is April 2026.
# Simplification: write off if age >= (2069 - year) in the base 2023 dataset.
# This matches the 25-year-from-first-repayment rule for the 1998-2011 cohort.
_PLAN_1_WRITEOFF_AGE = lambda year: 2069 - year


def _promote_to_plan(plan, income, weights, eligible_mask, target_weighted_millions, plan_label):
    """Promote the highest-income eligible NONE people to plan_label until the
    weighted total reaches target_weighted_millions. Returns updated plan array."""
    import numpy as np

    target = target_weighted_millions * 1e6
    current = weights[plan == plan_label].sum()
    delta = target - current
    if delta <= 0:
        return plan

    candidates = np.where(eligible_mask & (plan == "NONE"))[0]
    if len(candidates) == 0:
        return plan

    # Rank candidates by income descending — highest earners promoted first
    order = candidates[np.argsort(income[candidates])[::-1]]
    promoted = 0.0
    for i in order:
        if promoted >= delta:
            break
        plan[i] = plan_label
        promoted += weights[i]

    return plan


def _demote_from_plan(plan, income, weights, plan_label, target_weighted_millions):
    """Demote the lowest-income plan holders to NONE when the target falls
    (e.g. Plan 2 declining as loans are paid off post-2030)."""
    import numpy as np

    target = target_weighted_millions * 1e6
    current = weights[plan == plan_label].sum()
    delta = current - target
    if delta <= 0:
        return plan

    holders = np.where(plan == plan_label)[0]
    order = holders[np.argsort(income[holders])]  # lowest income first
    demoted = 0.0
    for i in order:
        if demoted >= delta:
            break
        plan[i] = "NONE"
        demoted += weights[i]

    return plan


def _roll_student_loan_plans(dataset, year, weights):
    """Advance student loan plan assignments to match forecast targets.

    - Plan 1: write off loans where age >= (2069 - year) in the base dataset,
      reflecting the 25-year loan term for the pre-2012 cohort.
    - Plan 2: promote/demote NONE people in the 2012-2022 age band by income
      rank to hit DfE-forecast total outstanding borrower targets.
    - Plan 5: promote NONE people in the post-2023 age band by income rank.
    - Plan 4 and Postgraduate: unchanged.
    """
    import numpy as np

    age = np.array(dataset.person["age"][:]).astype(int)
    income = np.array(dataset.person["employment_income"][:])
    plan = np.array(dataset.person["student_loan_plan"][:], dtype=object)

    # Plan 1: write off loans for cohort beyond 25-year term
    writeoff_age = _PLAN_1_WRITEOFF_AGE(year)
    plan[(plan == "PLAN_1") & (age >= writeoff_age)] = "NONE"

    if year in _PLAN_TARGETS:
        target_p2, target_p5 = _PLAN_TARGETS[year]

        # Plan 2: started uni 2012-2022, in year Y ages (Y-2004) to (Y-1994)
        p2_eligible = (age >= year - 2004) & (age <= year - 1994)
        plan = _promote_to_plan(plan, income, weights, p2_eligible, target_p2, "PLAN_2")
        plan = _demote_from_plan(plan, income, weights, "PLAN_2", target_p2)

        # Plan 5: started uni 2023+, in year Y ages 18 to (Y-2005)
        p5_eligible = (age >= 18) & (age <= year - 2005)
        plan = _promote_to_plan(plan, income, weights, p5_eligible, target_p5, "PLAN_5")

    dataset.person["student_loan_plan"] = plan
    return dataset


def uprate_dataset(dataset: UKSingleYearDataset, target_year=2034):
    import numpy as np

    dataset = dataset.copy()
    uprating_factors = pd.read_csv(STORAGE_FOLDER / "uprating_factors.csv")
    uprating_factors = uprating_factors.set_index("Variable")
    start_year = int(dataset.time_period)

    for table in dataset.tables:
        for variable in table.columns:
            if variable in uprating_factors.index:
                factor = (
                    uprating_factors.loc[variable, str(target_year)]
                    / uprating_factors.loc[variable, str(start_year)]
                )
                table[variable] *= factor

    dataset.time_period = target_year

    if "student_loan_plan" in dataset.person.columns:
        # Pre-compute person weights (household weight mapped to persons)
        person_hh_id = dataset.person["person_household_id"][:]
        hh_id = dataset.household["household_id"][:]
        hh_weight = dataset.household["household_weight"][:]
        weight_by_hh = dict(zip(hh_id, hh_weight))
        weights = np.array([weight_by_hh[i] for i in person_hh_id])

        for year in range(start_year + 1, target_year + 1):
            dataset = _roll_student_loan_plans(dataset, year, weights)

    return dataset


if __name__ == "__main__":
    create_policyengine_uprating_factors_table()
