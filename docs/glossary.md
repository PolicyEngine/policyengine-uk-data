# Glossary

## Datasets and Surveys

### FRS (Family Resources Survey)
The primary UK household survey conducted annually by the Department for Work and Pensions. Covers ~20,000 households with detailed information on demographics, income, benefits, and housing. The main data source for PolicyEngine UK Data.

### WAS (Wealth and Assets Survey)
Biennial ONS survey of ~20,000 households focusing on household wealth including property, financial assets, pensions, and debt. Used to impute wealth variables.

### LCFS (Living Costs and Food Survey)
Annual ONS survey of ~5,000 households recording detailed consumption expenditure. Used to impute consumption variables for VAT analysis.

### SPI (Survey of Personal Incomes)
HMRC administrative dataset based on tax records covering all UK taxpayers. A 1% sample (~300,000 individuals) is released for research. Used to correct high-income underreporting.

### ETB (Effects of Taxes and Benefits on Household Income)
ONS analysis based on LCFS data showing the redistributive effects of taxes and benefits. Used to impute VAT exposure rates.

## Dataset Variants

### ExtendedFRS
FRS enhanced with imputed wealth (from WAS) and consumption (from LCFS) variables. First enhancement stage.

### EnhancedFRS
ExtendedFRS with additional high-income enhancement using SPI data to correct income underreporting. Recommended for most analyses.

### ReweightedFRS
EnhancedFRS with calibrated weights to match 2000+ official statistics from HMRC, DWP, and ONS. Maximum accuracy variant.

## Statistical Terms

### Calibration
Process of adjusting survey weights to match known population totals or distributional targets. In PolicyEngine UK Data, weights are calibrated to match official statistics on demographics, incomes, and tax-benefit programs.

### Imputation
Statistical technique to estimate missing variables using machine learning models trained on other surveys. PolicyEngine uses Quantile Regression Forests for imputation.

### Microdata
Individual-level (person or household) data, as opposed to aggregated statistics. Enables detailed distributional analysis.

### Microsimulation
Modeling technique that applies policy rules to representative microdata to estimate policy impacts on individuals and the population.

### QRF (Quantile Regression Forests)
Machine learning algorithm that predicts the full conditional distribution of a variable, not just its mean. Used for imputation to preserve distributional properties.

### Reweighting
See Calibration.

## Entities

### Person
Individual in the dataset. Basic unit of analysis for many variables like age, gender, employment.

### Benefit Unit
Group of adults and children whose benefit entitlements are assessed together. Usually a family within a household.

### Household
Group of people living at the same address. May contain multiple benefit units (e.g., adult children living with parents).

## Income Concepts

### Gross Income
Total income before taxes and including benefits.

### Net Income
Income after taxes and National Insurance contributions, including benefits.

### Equivalised Income
Income adjusted for household size and composition to enable comparisons. Uses Modified OECD equivalence scale.

### Market Income
Income from employment, self-employment, investments, and pensions before taxes and benefits.

## Tax-Benefit System

### Universal Credit (UC)
Main means-tested benefit in the UK, replacing six legacy benefits. Combines support for unemployment, low income, housing costs, children, and disabilities.

### Income Tax
Progressive tax on income with multiple bands. Includes Personal Allowance (tax-free amount), Basic Rate (20%), Higher Rate (40%), and Additional Rate (45%).

### National Insurance (NI)
Social insurance contributions on earnings. Separate rates for employees, employers, and self-employed. Establishes eligibility for State Pension and other contributory benefits.

### VAT (Value Added Tax)
Consumption tax applied to most goods and services. Standard rate 20%, reduced rate 5%, zero rate for some essentials.

### Council Tax
Local property tax based on property value bands. Varies by local authority.

## Methodology Terms

### Enhancement
Process of improving FRS data by adding variables or correcting biases. Includes imputation (adding variables) and income correction (fixing underreporting).

### Loss Function
Metric used to evaluate dataset quality by comparing estimates to known statistics. Lower loss indicates better match to reality.

###

 Representative
Sample that accurately reflects the characteristics of the full population when appropriate weights are applied.

### Validation
Process of comparing dataset estimates against official statistics to assess accuracy.

### Weight
Multiplier applied to each household/person indicating how many real-world households/people they represent. Essential for population-level statistics.

## PolicyEngine Terms

### Reform
Change to policy parameters (e.g., tax rates, benefit amounts). Can be applied to simulations to estimate impacts.

### Simulation
Application of the tax-benefit model to a dataset to calculate taxes, benefits, and net incomes under current or reformed policy.

### Variable
Any measurable characteristic in the model (e.g., age, income, tax liability). Can be inputs (from data) or calculated (by model).

## UK Government Departments

### DWP (Department for Work and Pensions)
Responsible for welfare and pension policy. Publishes FRS and benefit statistics.

### HMRC (HM Revenue & Customs)
Tax authority. Publishes tax statistics and SPI data.

### ONS (Office for National Statistics)
National statistical institute. Publishes WAS, LCFS, ETB, and demographic statistics.

### OBR (Office for Budget Responsibility)
Independent fiscal watchdog. Publishes forecasts and policy costings used for validation.

## Research Terms

### Gini Coefficient
Measure of income inequality ranging from 0 (perfect equality) to 1 (perfect inequality). Commonly reported for income distributions.

### Poverty Rate
Percentage of population below a poverty threshold. UK typically uses 60% of median income (relative poverty) or inflation-adjusted threshold (absolute poverty).

### Decile
One-tenth of a distribution. First decile = bottom 10%, tenth decile = top 10%. Used to analyze distributional impacts.

### Marginal Tax Rate (MTR)
Percentage of an additional pound of income lost to taxes and benefit withdrawal. Can exceed 100% due to benefit tapers.

### Winners and Losers
Households gaining (winners) or losing (losers) income under a policy reform.

## Abbreviations

- **AGPL**: GNU Affero General Public License (software license)
- **API**: Application Programming Interface
- **CSV**: Comma-Separated Values (data format)
- **GCP**: Google Cloud Platform
- **HDF5**: Hierarchical Data Format 5 (efficient data storage)
- **ML**: Machine Learning
- **OECD**: Organisation for Economic Co-operation and Development
- **PyPI**: Python Package Index
- **UK**: United Kingdom

## See Also

- [Data Sources](data_sources.md) - Detailed information on each survey
- [Methodology](methodology.ipynb) - Technical details of enhancement process
- [API Reference](api_reference.md) - Function and class documentation