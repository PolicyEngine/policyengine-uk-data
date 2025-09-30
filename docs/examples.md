# Usage Examples

This page provides practical examples of using PolicyEngine UK Data for policy analysis.

## Basic Analysis

### Loading and Exploring a Dataset

```python
from policyengine_uk_data import EnhancedFRS_2022_23
from policyengine_uk import Microsimulation
import pandas as pd

# Load dataset and create simulation
simulation = Microsimulation(dataset=EnhancedFRS_2022_23)

# Get basic statistics
n_people = len(simulation.calculate("person_id", period=2025))
n_households = len(simulation.calculate("household_id", period=2025).unique())

print(f"Sample size: {n_people:,} people in {n_households:,} households")

# Calculate key aggregates
employment_income = simulation.calculate("employment_income", period=2025).sum() / 1e9
benefits = simulation.calculate("benefits", period=2025).sum() / 1e9
income_tax = simulation.calculate("income_tax", period=2025).sum() / 1e9

print(f"Employment income: £{employment_income:.1f}bn")
print(f"Benefits: £{benefits:.1f}bn")
print(f"Income tax: £{income_tax:.1f}bn")
```

### Comparing Datasets

```python
from policyengine_uk_data import FRS_2022_23, EnhancedFRS_2022_23

def get_income_stats(dataset):
    sim = Microsimulation(dataset=dataset)
    income = sim.calculate("household_net_income", period=2025)
    weights = sim.calculate("household_weight", period=2025)

    mean = (income * weights).sum() / weights.sum()
    total = income.sum() / 1e9

    return {"mean": mean, "total": total}

frs_stats = get_income_stats(FRS_2022_23)
efrs_stats = get_income_stats(EnhancedFRS_2022_23)

print(f"FRS mean income: £{frs_stats['mean']:,.0f}")
print(f"Enhanced FRS mean income: £{efrs_stats['mean']:,.0f}")
print(f"Difference: £{efrs_stats['mean'] - frs_stats['mean']:,.0f}")
```

## Policy Reform Analysis

### Simple Tax Change

```python
from policyengine_uk import Microsimulation, Reform
from policyengine_uk_data import EnhancedFRS_2022_23

# Define a basic rate threshold increase
class BasicRateIncrease(Reform):
    def apply(self):
        self.update_parameter(
            "gov.hmrc.income_tax.rates.uk[0].threshold",
            "2025-01-01.2099-12-31",
            15_000  # Increase from ~£12,570 to £15,000
        )

# Calculate impact
baseline = Microsimulation(dataset=EnhancedFRS_2022_23)
reformed = Microsimulation(dataset=EnhancedFRS_2022_23, reform=BasicRateIncrease)

# Revenue impact
baseline_revenue = baseline.calculate("income_tax", period=2025).sum()
reformed_revenue = reformed.calculate("income_tax", period=2025).sum()
revenue_change = (reformed_revenue - baseline_revenue) / 1e9

print(f"Revenue change: £{revenue_change:.2f}bn")

# Winners and losers
baseline_income = baseline.calculate("household_net_income", period=2025)
reformed_income = reformed.calculate("household_net_income", period=2025)
change = reformed_income - baseline_income

winners = (change > 0).sum()
losers = (change < 0).sum()
unchanged = (change == 0).sum()

print(f"Winners: {winners:,} households")
print(f"Losers: {losers:,} households")
print(f"Unchanged: {unchanged:,} households")
```

### Universal Basic Income

```python
class UniversalBasicIncome(Reform):
    def apply(self):
        # £100/week UBI for all adults
        self.update_parameter(
            "gov.contrib.ubi.adult.amount",
            "2025-01-01.2099-12-31",
            100 * 52  # Weekly to annual
        )

baseline = Microsimulation(dataset=EnhancedFRS_2022_23)
ubi_sim = Microsimulation(dataset=EnhancedFRS_2022_23, reform=UniversalBasicIncome)

# Cost
ubi_cost = ubi_sim.calculate("universal_basic_income", period=2025).sum() / 1e9
print(f"UBI cost: £{ubi_cost:.1f}bn/year")

# Poverty impact
baseline_poverty = (
    baseline.calculate("in_absolute_poverty", period=2025).sum()
)
ubi_poverty = (
    ubi_sim.calculate("in_absolute_poverty", period=2025).sum()
)

print(f"Poverty reduction: {baseline_poverty - ubi_poverty:,} people")
```

## Distributional Analysis

### Income Deciles

```python
import numpy as np
import pandas as pd

simulation = Microsimulation(dataset=EnhancedFRS_2022_23)

# Get household data
income = simulation.calculate("household_net_income", period=2025)
weights = simulation.calculate("household_weight", period=2025)

# Calculate deciles
decile = simulation.calculate("household_income_decile", period=2025)

# Mean income by decile
decile_data = pd.DataFrame({
    "income": income,
    "weight": weights,
    "decile": decile
})

decile_means = decile_data.groupby("decile").apply(
    lambda x: (x.income * x.weight).sum() / x.weight.sum()
)

print("Mean income by decile:")
for d, mean in decile_means.items():
    print(f"  Decile {d}: £{mean:,.0f}")
```

### Gini Coefficient

```python
def gini(values, weights):
    """Calculate Gini coefficient."""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumsum = np.cumsum(sorted_weights)
    cumsum_values = np.cumsum(sorted_values * sorted_weights)

    return (
        (2 * np.sum(cumsum * sorted_values * sorted_weights)) /
        (cumsum[-1] * cumsum_values[-1]) - 1
    )

simulation = Microsimulation(dataset=EnhancedFRS_2022_23)
income = simulation.calculate("household_net_income", period=2025)
weights = simulation.calculate("household_weight", period=2025)

gini_coef = gini(income, weights)
print(f"Gini coefficient: {gini_coef:.3f}")
```

## Regional Analysis

### Income by Region

```python
simulation = Microsimulation(dataset=EnhancedFRS_2022_23)

income = simulation.calculate("household_net_income", period=2025)
region = simulation.calculate("region", period=2025)
weights = simulation.calculate("household_weight", period=2025)

region_income = pd.DataFrame({
    "income": income,
    "region": region,
    "weight": weights
})

regional_means = region_income.groupby("region").apply(
    lambda x: (x.income * x.weight).sum() / x.weight.sum()
)

print("Mean household income by region:")
for r, mean in regional_means.items():
    print(f"  {r}: £{mean:,.0f}")
```

## Custom Analysis

### Targeting Analysis

```python
# Analyze take-up of a benefit
simulation = Microsimulation(dataset=EnhancedFRS_2022_23)

# Eligible population
eligible = simulation.calculate("universal_credit_entitlement", period=2025) > 0

# Actual recipients
receiving = simulation.calculate("universal_credit", period=2025) > 0

# Take-up rate
takeup_rate = receiving[eligible].mean()
print(f"Universal Credit take-up rate: {takeup_rate:.1%}")
```

### Marginal Tax Rates

```python
def marginal_tax_rate(simulation, person_id, base_earnings):
    """Calculate marginal tax rate for a person."""
    # Baseline
    base_net = simulation.calculate("net_income", period=2025)[person_id]

    # Increment earnings by £1000
    simulation.set_input("employment_income", period=2025,
                         {person_id: base_earnings + 1000})
    new_net = simulation.calculate("net_income", period=2025)[person_id]

    # MTR = 1 - (change in net / change in gross)
    mtr = 1 - (new_net - base_net) / 1000
    return mtr

simulation = Microsimulation(dataset=EnhancedFRS_2022_23)
# Calculate MTRs for employed people
employment_income = simulation.calculate("employment_income", period=2025)
employed = employment_income > 0

mtrs = [
    marginal_tax_rate(simulation, pid, employment_income[pid])
    for pid in range(len(employed)) if employed[pid]
]

print(f"Mean MTR for employed: {np.mean(mtrs):.1%}")
```

## Validation and Quality Checks

### Compare to Official Statistics

```python
from policyengine_uk_data.utils import get_loss_results

results = get_loss_results(EnhancedFRS_2022_23, 2025)

# Filter to specific statistics
tax_stats = results[results.name.str.contains("obr")]
print("Tax-benefit program accuracy:")
print(tax_stats[["name", "target", "estimate", "abs_rel_error"]].head(10))

# Overall accuracy
print(f"\nMean absolute relative error: {results.abs_rel_error.mean():.2%}")
print(f"Median absolute relative error: {results.abs_rel_error.median():.2%}")
```

## Tips and Best Practices

1. **Cache simulations** when running multiple reforms on the same baseline
2. **Use vectorized operations** instead of loops for better performance
3. **Check validation metrics** to understand dataset accuracy for your use case
4. **Start with EnhancedFRS** unless you have specific reasons to use another variant
5. **Weight all statistics** using household/person weights for population estimates

## Next Steps

- [API Reference](api-reference.md) - Complete function documentation
- [Methodology](methodology.ipynb) - Understand dataset construction
- [Validation](validation/) - See accuracy metrics