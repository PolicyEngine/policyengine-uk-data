# Introduction

Welcome to PolicyEngine UK Data - a comprehensive solution for creating representative microdata for United Kingdom policy analysis.

## What is PolicyEngine UK Data?

PolicyEngine UK Data transforms the UK Family Resources Survey into enhanced microdata suitable for accurate tax-benefit policy analysis. By combining multiple government surveys and applying advanced statistical techniques, we create datasets that accurately represent the UK population's demographics, incomes, wealth, and consumption patterns.

## The Challenge

Effective tax-benefit policy analysis requires:

1. **An accurate model of policy rules** - How do taxes and benefits actually work?
2. **Accurate representation of the population** - Who are the people affected by these policies?

PolicyEngine UK provides the first (the tax-benefit model). This package provides the second (the microdata).

The challenge is that no single survey captures everything we need:
- The FRS has good demographics but underreports income and lacks wealth data
- The WAS has wealth but smaller sample sizes
- The SPI has accurate high incomes but no demographics
- The LCFS has consumption but only ~5,000 households

## Our Solution

We combine the strengths of multiple surveys:

```
FRS (demographics) + WAS (wealth) + LCFS (consumption) + SPI (high incomes)
                                ↓
                    Statistical enhancement
                                ↓
                    Calibration to match
                    official statistics
                                ↓
                Enhanced representative microdata
```

The result is a dataset that:
- ✅ Matches official HMRC, DWP, and ONS statistics
- ✅ Includes wealth and consumption variables
- ✅ Correctly represents high-income individuals
- ✅ Enables accurate policy impact analysis

## Who Should Use This?

### Researchers
- Academic economists studying UK tax-benefit policy
- Policy researchers analyzing distributional impacts
- PhD students modeling fiscal reforms

### Policy Analysts
- Government departments evaluating policy options
- Think tanks developing policy proposals
- Advocacy organizations assessing policy impacts

### Data Scientists
- Building tax-benefit calculators
- Developing distributional analysis tools
- Creating policy simulation platforms

## What You Can Do

With PolicyEngine UK Data, you can:

- **Estimate policy costs** - How much would a reform cost or save?
- **Analyze distributional impacts** - Who wins and loses from policy changes?
- **Calculate poverty and inequality** - How do policies affect poverty rates?
- **Model benefit take-up** - How many people are eligible vs. receiving benefits?
- **Regional analysis** - How do impacts vary by constituency or local authority?
- **Behavioral responses** - How might people respond to policy incentives?

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Install and use the package | [Getting Started](getting-started.md) |
| See code examples | [Examples](examples.md) |
| Understand the methodology | [Methodology](methodology.ipynb) |
| Look up functions and classes | [API Reference](api-reference.md) |
| Check dataset accuracy | [Validation](validation/) |
| Understand technical terms | [Glossary](glossary.md) |
| Learn about data sources | [Data Sources](data-sources.md) |

## How This Documentation is Organized

1. **User Guide** - Practical information for using the package
   - [Getting Started](getting-started.md) - Installation and first steps
   - [Examples](examples.md) - Code examples for common tasks
   - [API Reference](api-reference.md) - Complete function documentation
   - [Data Sources](data-sources.md) - Information on source surveys
   - [Glossary](glossary.md) - Definitions and terminology

2. **Technical Details** - In-depth methodology
   - [Methodology](methodology.ipynb) - Step-by-step dataset creation
   - [Pension Contributions](pension_contributions.ipynb) - Pension data processing
   - [Constituency Methodology](constituency_methodology.ipynb) - Constituency-level datasets
   - [Local Authority Methodology](LA_methodology.ipynb) - Local authority datasets

3. **Validation** - Accuracy and quality assurance
   - [National Validation](validation/national.ipynb) - Comparison to national statistics
   - [Constituency Validation](validation/constituencies.ipynb) - Constituency-level accuracy
   - [Local Authority Validation](validation/local_authorities.ipynb) - Local authority accuracy

## Project Context

PolicyEngine UK Data is part of the broader PolicyEngine ecosystem:

- **[PolicyEngine UK](https://github.com/PolicyEngine/policyengine-uk)** - The tax-benefit microsimulation model
- **[PolicyEngine](https://policyengine.org)** - Web application for policy analysis
- **[PolicyEngine US Data](https://github.com/PolicyEngine/policyengine-us-data)** - Equivalent dataset for the United States

## Contributing

We welcome contributions! Whether you're fixing bugs, improving documentation, or adding features, please see our [GitHub repository](https://github.com/PolicyEngine/policyengine-uk-data) to get started.

## License and Citation

PolicyEngine UK Data is open source (AGPL-3.0). If you use it in research, please cite:

```
PolicyEngine. (2024). PolicyEngine UK Data.
https://github.com/PolicyEngine/policyengine-uk-data
```

For methodology details, see our [Methodology page](methodology.ipynb).

