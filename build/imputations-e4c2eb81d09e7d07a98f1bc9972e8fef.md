# Imputations

PolicyEngine UK Data enhances the Family Resources Survey with variables from other surveys using statistical imputation. All imputations use **Quantile Regression Forests (QRF)**, which predict the full conditional distribution of target variables given predictor variables.

## Imputation Pipeline Order

The imputations are applied in this order (dependencies noted):

1. **Wealth** (from WAS)
2. **Consumption** (from LCFS) — requires `num_vehicles` from wealth
3. **VAT** (from ETB)
4. **Public Services** (from ETB)
5. **Income** (from SPI)
6. **Capital Gains** (from Advani-Summers data)
7. **Salary Sacrifice** (from FRS subsample)
8. **Student Loan Plan** (rule-based, from age)

---

## Wealth Imputation

**Source:** Wealth and Assets Survey (WAS) Round 7 (2018-2020)

Imputes household wealth components using demographic and income predictors.

### Predictors
| Variable | Description |
|----------|-------------|
| `household_net_income` | Total household income after taxes |
| `num_adults` | Number of adults in household |
| `num_children` | Number of children in household |
| `private_pension_income` | Income from private pensions |
| `employment_income` | Income from employment |
| `self_employment_income` | Income from self-employment |
| `capital_income` | Income from capital/investments |
| `num_bedrooms` | Number of bedrooms in dwelling |
| `council_tax` | Annual council tax payment |
| `is_renting` | Whether household rents (vs owns) |
| `region` | UK region |

### Outputs
| Variable | Description |
|----------|-------------|
| `owned_land` | Value of owned land |
| `property_wealth` | Total property wealth |
| `corporate_wealth` | Shares, pensions, investment ISAs |
| `gross_financial_wealth` | Total financial assets |
| `net_financial_wealth` | Financial assets minus liabilities |
| `main_residence_value` | Value of main home |
| `other_residential_property_value` | Value of other properties |
| `non_residential_property_value` | Value of non-residential property |
| `savings` | Savings account balances |
| `num_vehicles` | Number of vehicles owned |

---

## Consumption Imputation

**Source:** Living Costs and Food Survey (LCFS) 2021-22

Imputes household spending patterns for indirect tax modeling.

### Predictors
| Variable | Description |
|----------|-------------|
| `is_adult` | Number of adults |
| `is_child` | Number of children |
| `region` | UK region |
| `employment_income` | Employment income |
| `self_employment_income` | Self-employment income |
| `private_pension_income` | Private pension income |
| `household_net_income` | Total household income |
| `has_fuel_consumption` | Whether household buys petrol/diesel (from WAS) |

### Outputs
| Variable | Description |
|----------|-------------|
| `food_and_non_alcoholic_beverages_consumption` | Food spending |
| `alcohol_and_tobacco_consumption` | Alcohol/tobacco spending |
| `clothing_and_footwear_consumption` | Clothing spending |
| `housing_water_and_electricity_consumption` | Housing costs |
| `household_furnishings_consumption` | Furnishings spending |
| `health_consumption` | Health spending |
| `transport_consumption` | Transport spending |
| `communication_consumption` | Communication spending |
| `recreation_consumption` | Recreation spending |
| `education_consumption` | Education spending |
| `restaurants_and_hotels_consumption` | Restaurants/hotels spending |
| `miscellaneous_consumption` | Other spending |
| `petrol_spending` | Petrol fuel spending |
| `diesel_spending` | Diesel fuel spending |
| `domestic_energy_consumption` | Home energy spending |

### Bridging WAS Vehicle Ownership to LCFS Fuel Spending

LCFS 2-week diaries undercount fuel purchasers (58%) compared to actual vehicle ownership (78% per NTS 2024). We bridge this gap using WAS vehicle data:

1. **In WAS**: Create `has_fuel_consumption` from vehicle ownership:
   - `has_fuel = (num_vehicles > 0) AND (random < 0.90)`
   - The 90% accounts for EVs/PHEVs that don't buy petrol/diesel
   - Source: NTS 2024 shows 59% petrol + 30% diesel + ~1% hybrid fuel use

2. **Train QRF**: Predict `has_fuel_consumption` from demographics (income, adults, children, region)

3. **Apply to LCFS**: Impute `has_fuel_consumption` to LCFS households before training consumption model

4. **At FRS imputation time**: Compute `has_fuel_consumption` directly from `num_vehicles` (already calibrated to NTS targets)

This ensures fuel duty incidence aligns with actual vehicle ownership (~70% of households = 78% vehicles × 90% ICE) rather than LCFS diary randomness.

---

## VAT Imputation

**Source:** Effects of Taxes and Benefits (ETB) 1977-2021

Imputes the share of household spending subject to full-rate VAT.

### Predictors
| Variable | Description |
|----------|-------------|
| `is_adult` | Number of adults |
| `is_child` | Number of children |
| `is_SP_age` | Number at State Pension age |
| `household_net_income` | Total household income |

### Outputs
| Variable | Description |
|----------|-------------|
| `full_rate_vat_expenditure_rate` | Share of spending at 20% VAT |

---

## Income Imputation

**Source:** Survey of Personal Incomes (SPI) 2020-21

Imputes detailed income components to create "synthetic taxpayers" with higher incomes than typically captured in the FRS. These records initially have zero weight but can be upweighted during calibration to match HMRC income distribution targets.

### Predictors
| Variable | Description |
|----------|-------------|
| `age` | Person's age |
| `gender` | Male/Female |
| `region` | UK region |

### Outputs
| Variable | Description |
|----------|-------------|
| `employment_income` | Income from employment |
| `self_employment_income` | Income from self-employment |
| `savings_interest_income` | Interest on savings |
| `dividend_income` | Dividend income |
| `private_pension_income` | Private pension income |
| `property_income` | Rental/property income |

---

## Capital Gains Imputation

**Source:** Advani-Summers capital gains distribution data

Uses a gradient-based optimization approach rather than QRF. The dataset is doubled, with one half receiving imputed capital gains amounts. Weights are then optimized to match the empirical relationship between total income and capital gains incidence.

### Method
1. Double the dataset (original + clone)
2. Assign capital gains to one adult per household in the cloned half
3. Optimize blend weights to match income-band capital gains incidence from Advani-Summers data

---

## Salary Sacrifice Imputation

**Source:** FRS 2023-24 (respondents asked about salary sacrifice)

Imputes pension contributions made via salary sacrifice arrangements.

### Predictors
| Variable | Description |
|----------|-------------|
| `age` | Person's age |
| `employment_income` | Employment income |

### Outputs
| Variable | Description |
|----------|-------------|
| `pension_contributions_via_salary_sacrifice` | Annual SS pension contributions |

### Training Data
- FRS respondents with `SALSAC='1'` (Yes): ~224 jobs with reported amounts
- FRS respondents with `SALSAC='2'` (No): ~3,803 jobs with 0
- Imputation candidates (`SALSAC=' '`): ~13,265 jobs

---

## Student Loan Plan Imputation

**Source:** Rule-based (not QRF)

Assigns student loan plan type based on age and reported repayments.

### Logic
1. If `student_loan_repayments > 0`, person has a loan
2. Estimate university start year = `simulation_year - age + 18`
3. Assign plan:
   - **Plan 1**: Started before September 2012
   - **Plan 2**: Started September 2012 - August 2023
   - **Plan 5**: Started September 2023 onwards

---

## Calibration Targets

After imputation, household weights are calibrated to match aggregate statistics from:

| Source | Targets |
|--------|---------|
| **OBR** | Tax revenues, benefit expenditures (20 programs) |
| **ONS** | Age/region populations, family types, tenure |
| **HMRC** | Income distributions by band (7 income types × 14 bands) |
| **DWP** | Universal Credit statistics, two-child limit |
| **NTS** | Vehicle ownership (22% none, 44% one, 34% two+) |
| **Council Tax** | Households by council tax band |
