# PolicyEngine UK Data

[![Documentation](https://img.shields.io/badge/docs-live-blue)](https://policyengine.github.io/policyengine-uk-data/)
[![PyPI version](https://badge.fury.io/py/policyengine-uk-data.svg)](https://badge.fury.io/py/policyengine-uk-data)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

**PolicyEngine UK Data** creates representative microdata for the United Kingdom, designed for use in the [PolicyEngine UK](https://github.com/PolicyEngine/policyengine-uk) tax-benefit microsimulation model.

## What is this?

This package transforms the UK Family Resources Survey (FRS) into an enhanced dataset suitable for accurate tax-benefit policy analysis. The enhancement process includes:

- **Imputation** of missing variables (wealth, consumption, VAT exposure)
- **Income enhancement** using Survey of Personal Incomes (SPI) data
- **Calibration** to match official statistics from HMRC, DWP, and ONS
- **Local area datasets** for constituencies and local authorities

The result is a dataset that accurately represents the UK population and economy, enabling precise policy impact analysis.

## Installation

### Prerequisites

- Python 3.13 or higher
- [Hugging Face account](https://huggingface.co/) (for data downloads)

### Install from PyPI

```bash
pip install policyengine-uk-data
```

### Install from source

```bash
git clone https://github.com/PolicyEngine/policyengine-uk-data.git
cd policyengine-uk-data
pip install -e ".[dev]"
```

### Authentication

Set your Hugging Face token as an environment variable:

```bash
export HUGGING_FACE_TOKEN="your_token_here"
```

Or create a `.env` file in your project root:

```
HUGGING_FACE_TOKEN=your_token_here
```

## Quick Start

```python
from policyengine_uk_data import EnhancedFRS_2022_23
from policyengine_uk import Microsimulation

# Load the enhanced dataset
dataset = EnhancedFRS_2022_23

# Create a microsimulation for 2025
simulation = Microsimulation(dataset=dataset)

# Calculate total employment income
employment_income = simulation.calculate("employment_income", period=2025)
print(f"Total employment income: £{employment_income.sum() / 1e9:.1f}bn")

# Calculate mean household income
household_income = simulation.calculate("household_net_income", period=2025)
weights = simulation.calculate("household_weight", period=2025)
mean_income = (household_income * weights).sum() / weights.sum()
print(f"Mean household net income: £{mean_income:,.0f}")
```

## Available Datasets

| Dataset | Description | Use Case |
|---------|-------------|----------|
| `FRS_2022_23` | Raw FRS with benefits as reported | Baseline comparison |
| `ExtendedFRS_2022_23` | FRS + imputed wealth/consumption | Basic policy analysis |
| `EnhancedFRS_2022_23` | Extended + SPI income enhancement | Recommended for most analyses |
| `ReweightedFRS_2022_23` | Enhanced + calibrated weights | Maximum accuracy |

## Documentation

- **[Getting Started Guide](https://policyengine.github.io/policyengine-uk-data/getting_started.html)** - Detailed installation and setup
- **[Methodology](https://policyengine.github.io/policyengine-uk-data/methodology.html)** - How we create the datasets
- **[API Reference](https://policyengine.github.io/policyengine-uk-data/api_reference.html)** - Complete API documentation
- **[Examples](https://policyengine.github.io/policyengine-uk-data/examples.html)** - Usage examples and tutorials
- **[Validation](https://policyengine.github.io/policyengine-uk-data/validation/)** - Comparison with official statistics

## Building the Datasets

To rebuild the datasets from source data:

```bash
# Download prerequisites (requires authentication)
make download

# Build all datasets
make data

# Run tests
make test

# Upload to storage (requires GCP credentials)
make upload
```

## Data Sources

This package combines data from multiple UK surveys:

- **Family Resources Survey (FRS)** - household demographics, income, benefits
- **Wealth and Assets Survey (WAS)** - wealth imputations
- **Living Costs and Food Survey (LCFS)** - consumption imputations
- **Survey of Personal Incomes (SPI)** - high-income enhancement
- **Effects of Taxes and Benefits (ETB)** - VAT exposure

See [Data Sources documentation](https://policyengine.github.io/policyengine-uk-data/data_sources.html) for details.

## Citation

If you use this package in research, please cite:

```
PolicyEngine. (2024). PolicyEngine UK Data. GitHub.
https://github.com/PolicyEngine/policyengine-uk-data
```

For the methodology, see our [documentation](https://policyengine.github.io/policyengine-uk-data/methodology.html).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://policyengine.github.io/policyengine-uk-data/
- **Issues**: https://github.com/PolicyEngine/policyengine-uk-data/issues
- **Discussions**: https://github.com/PolicyEngine/policyengine-uk-data/discussions
- **Email**: hello@policyengine.org

## Related Projects

- [**PolicyEngine UK**](https://github.com/PolicyEngine/policyengine-uk) - UK tax-benefit microsimulation model
- [**PolicyEngine**](https://github.com/PolicyEngine/policyengine) - Policy simulation platform
- [**PolicyEngine US Data**](https://github.com/PolicyEngine/policyengine-us-data) - US equivalent dataset