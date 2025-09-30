# Getting Started

This guide will help you install and start using PolicyEngine UK Data.

## Prerequisites

Before installing, ensure you have:

1. **Python 3.13 or higher**
   ```bash
   python --version  # Should be 3.13+
   ```

2. **A Hugging Face account**
   - Sign up at [huggingface.co](https://huggingface.co/)
   - Create an access token at [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - The token needs **read** access

3. **(Optional) Google Cloud credentials**
   - Only needed if you're building datasets from scratch
   - For most users, pre-built datasets are available via Hugging Face

## Installation

### Standard Installation

Install from PyPI:

```bash
pip install policyengine-uk-data
```

### Development Installation

For contributing or building datasets:

```bash
# Clone the repository
git clone https://github.com/PolicyEngine/policyengine-uk-data.git
cd policyengine-uk-data

# Install with development dependencies
pip install -e ".[dev]"
```

## Authentication

### Hugging Face Token

Set your Hugging Face token as an environment variable:

**Linux/macOS:**
```bash
export HUGGING_FACE_TOKEN="your_token_here"
```

**Windows (Command Prompt):**
```cmd
set HUGGING_FACE_TOKEN=your_token_here
```

**Windows (PowerShell):**
```powershell
$env:HUGGING_FACE_TOKEN="your_token_here"
```

**Or use a `.env` file:**

Create a `.env` file in your project directory:

```
HUGGING_FACE_TOKEN=your_token_here
```

The package will automatically load environment variables from `.env` files.

## First Steps

### 1. Import and Load a Dataset

```python
from policyengine_uk_data import EnhancedFRS_2022_23

# The dataset will download automatically on first use
dataset = EnhancedFRS_2022_23
```

First-time downloads may take a few minutes depending on your connection.

### 2. Create a Microsimulation

```python
from policyengine_uk import Microsimulation

# Create a simulation for 2025
simulation = Microsimulation(dataset=dataset)
```

### 3. Calculate Variables

```python
# Calculate employment income for all persons
employment_income = simulation.calculate("employment_income", period=2025)

# Calculate household net income
household_income = simulation.calculate("household_net_income", period=2025)

# Get household weights for population-representative statistics
weights = simulation.calculate("household_weight", period=2025)
```

### 4. Compute Aggregate Statistics

```python
import numpy as np

# Total employment income (in billions)
total_employment = employment_income.sum() / 1e9
print(f"Total employment income: £{total_employment:.1f}bn")

# Mean household income
mean_income = (household_income * weights).sum() / weights.sum()
print(f"Mean household net income: £{mean_income:,.0f}")

# Median household income
sorted_indices = np.argsort(household_income)
cumsum = np.cumsum(weights[sorted_indices])
median_index = sorted_indices[np.searchsorted(cumsum, cumsum[-1] / 2)]
median_income = household_income[median_index]
print(f"Median household net income: £{median_income:,.0f}")
```

## Choosing a Dataset

PolicyEngine UK Data provides four dataset variants:

| Dataset | When to Use | Pros | Cons |
|---------|-------------|------|------|
| `FRS_2022_23` | Comparing with raw FRS | Matches official FRS | Missing wealth/consumption, income underreporting |
| `ExtendedFRS_2022_23` | Basic analysis with wealth/consumption | Adds wealth and consumption variables | Still has income underreporting |
| `EnhancedFRS_2022_23` | **Most analyses** (recommended) | Corrects income distribution, adds wealth/consumption | Small dataset size increase |
| `ReweightedFRS_2022_23` | Maximum accuracy needed | Calibrated to match official statistics exactly | Slightly higher memory usage |

For most policy analysis, use `EnhancedFRS_2022_23`.

## Common Patterns

### Analyzing a Policy Reform

```python
from policyengine_uk import Microsimulation, Reform
from policyengine_uk_data import EnhancedFRS_2022_23

# Baseline simulation
baseline = Microsimulation(dataset=EnhancedFRS_2022_23)

# Define a reform (e.g., increase basic rate threshold)
class IncomeT

axReform(Reform):
    def apply(self):
        self.update_parameter("gov.hmrc.income_tax.rates.uk[0].threshold", "2025-01-01.2099-12-31", 15_000)

# Reformed simulation
reformed = Microsimulation(reform=IncomeT

axReform, dataset=EnhancedFRS_2022_23)

# Compare tax revenues
baseline_tax = baseline.calculate("income_tax", period=2025).sum()
reformed_tax = reformed.calculate("income_tax", period=2025).sum()

revenue_change = (reformed_tax - baseline_tax) / 1e9
print(f"Revenue change: £{revenue_change:.1f}bn")
```

### Working with Local Areas

```python
from policyengine_uk_data.datasets.local_areas import (
    Constituency_2024_25,
    LocalAuthority_2024_25
)

# Load constituency-level data
constituency_data = Constituency_2024_25
simulation = Microsimulation(dataset=constituency_data)

# Get constituency codes
constituency_codes = simulation.calculate("constituency", period=2025)

# Calculate statistics by constituency
# (Implementation depends on your specific needs)
```

## Troubleshooting

### Import Error: "Prerequisites not found"

The package checks for required data files on import. If you see this error:

```python
from policyengine_uk_data import check_prerequisites, download_prerequisites

# Check what's missing
check_prerequisites()

# Download missing files
download_prerequisites()
```

### Download Fails

If downloads fail:

1. **Check your Hugging Face token** is set correctly
2. **Check internet connection**
3. **Try clearing the cache:**
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

### Memory Issues

For large-scale analysis:

1. **Use a subset of the data:**
   ```python
   # Sample 10% of households
   simulation.sample_size = 0.1
   ```

2. **Calculate variables individually** rather than all at once

3. **Use `ReweightedFRS_2022_23`** instead of building custom datasets

### Performance Tips

- **Cache simulations** when running multiple reforms
- **Use vectorized operations** instead of loops
- **Profile your code** with `cProfile` to find bottlenecks
- **Consider using Dask** for truly large-scale analysis

## Next Steps

- **[Examples](examples.md)** - More detailed usage examples
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Methodology](methodology.ipynb)** - Understand how datasets are created
- **[Validation](validation/)** - See how datasets compare to official statistics

## Getting Help

- **Documentation**: [policyengine.github.io/policyengine-uk-data](https://policyengine.github.io/policyengine-uk-data/)
- **Issues**: [GitHub Issues](https://github.com/PolicyEngine/policyengine-uk-data/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PolicyEngine/policyengine-uk-data/discussions)
- **Email**: hello@policyengine.org