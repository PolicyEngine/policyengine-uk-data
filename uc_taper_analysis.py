"""
Analyse how changing the Universal Credit taper rate affects different
hypothetical households.

Uses policyengine_uk_compiled only for baseline parameter values, then
computes UC entitlement analytically:

    UC = max(0, max_UC - taper_rate * max(0, earnings - work_allowance))
"""

from policyengine_uk_compiled import Simulation

params = Simulation(year=2025).get_baseline_params()
uc = params["universal_credit"]

# Household archetypes: (label, monthly_max_UC, work_allowance, earnings_range)
# Work allowance: higher if no housing element, lower otherwise.
# We assume all receive housing element (lower work allowance).
households = [
    {
        "label": "Single adult, no children, over 25",
        "max_uc": uc["standard_allowance_single_over25"],
        "work_allowance": 0,  # no children/LCWRA → no work allowance
        "earnings": list(range(0, 3001, 100)),
    },
    {
        "label": "Single parent, 1 child, over 25",
        "max_uc": uc["standard_allowance_single_over25"] + uc["child_element_first"],
        "work_allowance": uc["work_allowance_lower"],
        "earnings": list(range(0, 3001, 100)),
    },
    {
        "label": "Couple, 2 children, over 25",
        "max_uc": (
            uc["standard_allowance_couple_over25"]
            + uc["child_element_first"]
            + uc["child_element_subsequent"]
        ),
        "work_allowance": uc["work_allowance_lower"],
        "earnings": list(range(0, 4001, 100)),
    },
    {
        "label": "Single adult with LCWRA, over 25",
        "max_uc": uc["standard_allowance_single_over25"] + uc["lcwra_element"],
        "work_allowance": uc["work_allowance_lower"],
        "earnings": list(range(0, 3001, 100)),
    },
]

TAPER_RATES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
CURRENT_TAPER = 0.55


def calc_uc(max_uc, work_allowance, earnings, taper_rate):
    return max(0, max_uc - taper_rate * max(0, earnings - work_allowance))


print("UC taper rate analysis — hypothetical households (2025/26)\n")
print(f"Current taper rate: {CURRENT_TAPER:.0%}")
print(f"Baseline parameters from policyengine_uk_compiled\n")

for hh in households:
    print(f"{'=' * 72}")
    print(f"  {hh['label']}")
    print(f"  Max UC: £{hh['max_uc']:.2f}/month | Work allowance: £{hh['work_allowance']:.0f}/month")
    print(f"{'=' * 72}")

    # Find key earnings thresholds for each taper rate
    print(f"\n  {'Earnings':>10}", end="")
    for rate in TAPER_RATES:
        marker = " *" if rate == CURRENT_TAPER else "  "
        print(f"  {rate:.0%}{marker:2}", end="")
    print()
    print(f"  {'-' * 10}", end="")
    for _ in TAPER_RATES:
        print(f"  {'------':>8}", end="")
    print()

    for e in hh["earnings"]:
        # Only print rows where at least one taper rate gives UC > 0
        # or it's a round £500
        ucs = [calc_uc(hh["max_uc"], hh["work_allowance"], e, r) for r in TAPER_RATES]
        if max(ucs) == 0 and e > 0:
            continue
        if e % 500 != 0 and e % 250 != 0:
            continue

        print(f"  £{e:>8,}", end="")
        for uc_val in ucs:
            print(f"  £{uc_val:>6.0f}", end="")
        print()

    # Summary: breakeven earnings (where UC hits zero)
    print(f"\n  Breakeven earnings (UC = £0):")
    for rate in TAPER_RATES:
        if hh["work_allowance"] > 0 or rate > 0:
            breakeven = hh["work_allowance"] + hh["max_uc"] / rate
        else:
            breakeven = float("inf")
        marker = " ← current" if rate == CURRENT_TAPER else ""
        print(f"    {rate:.0%} taper: £{breakeven:,.0f}/month (£{breakeven * 12:,.0f}/year){marker}")

    # Effective marginal tax rate in taper region
    # (taper rate IS the benefit withdrawal rate; combined with income tax + NI)
    it = params["income_tax"]
    ni = params["national_insurance"]
    basic_rate = it["uk_brackets"][0]["rate"]
    ni_rate = ni["main_rate"]

    print(f"\n  Effective marginal rate in taper region (basic-rate taxpayer):")
    print(f"    Income tax: {basic_rate:.0%} + NI: {ni_rate:.0%} + UC taper")
    for rate in TAPER_RATES:
        # UC taper applies to net earnings but simplified here as gross
        combined = basic_rate + ni_rate + rate * (1 - basic_rate - ni_rate)
        marker = " ← current" if rate == CURRENT_TAPER else ""
        print(f"    {rate:.0%} taper → {combined:.1%} combined EMTR{marker}")

    print()
