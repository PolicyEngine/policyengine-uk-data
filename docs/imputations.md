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

**Source:** Wealth and Assets Survey (WAS) Round 8 (April 2020 to March 2022), household file

Imputes household wealth components using demographic and income predictors. The targets are imputed as a chain: each is fitted on the predictors and every earlier target.

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
| Variable | WAS round 8 source | Description |
|----------|--------------------|-------------|
| `owned_land` | `DVLUKValR8_sum` | Value of land in the UK |
| `property_wealth` | `DVPropertyR8` | Gross value of all property, main residence included |
| `private_pension_wealth` | `totalpenr8_aggr` − `dvvaldbt_scaper8_aggr` | Private pension wealth other than current-employment defined benefit rights: defined contribution pots, retained rights, pensions in payment |
| `directly_held_shares` | `DVFShUKVR8_aggr` + `DVFESHARESR8_aggr` | UK shares and employee shares and options, held outside ISAs and pooled funds |
| `unit_and_investment_trusts` | `DVFCollVR8_aggr` | Unit trusts and investment trusts (one survey question) |
| `stocks_and_shares_isa` | `DVIISAVR8_aggr` | Stocks and shares (investment) ISAs |
| `corporate_wealth` | derived | `directly_held_shares` + `unit_and_investment_trusts` + `stocks_and_shares_isa`, summed after imputation |
| `gross_financial_wealth` | `HFINWR8_SUM` | Total financial assets |
| `net_financial_wealth` | `HFINWNTR8_Sum` | Financial assets minus liabilities |
| `main_residence_value` | `DVHValueR8` | Value of main home |
| `other_residential_property_value` | `DVHseValR8_sum` + `DVBltValR8_sum` | Second homes and buy-to-let property |
| `non_residential_property_value` | `DVBlDValR8_sum` | Other buildings, such as shops and garages |
| `savings` | `DVSaValR8_aggr` | Savings account balances (cash ISAs excluded) |
| `num_vehicles` | `vcarnr8` | Number of vehicles owned |
| `student_loan_balance` | `Tot_LosR8_aggr` − `Tot_los_exc_SLCR8_aggr` | Student Loans Company balance, allocated to people |
| `cash_isa` | `DVCISAVR8_aggr` | Cash ISAs |
| `other_residential_property_secured_debt` | `DVHseDebtR8_sum` + `DVBLtDebtR8_sum` | Mortgages and loans secured on second homes and buy-to-let property |
| `non_residential_property_secured_debt` | `DVBldDebtR8_sum` | Mortgages and loans secured on other buildings |
| `owned_land_secured_debt` | `DVLUKDebtR8_sum` | Loans secured on UK land |

`corporate_wealth` is the exact sum of its three components on every household; private pension wealth is not part of it, because pension rights are disregarded capital in every means test. A secured debt is zero wherever the household holds none of the asset it is secured on. Land and property overseas (`DVLOSValR8_sum`), other real estate (`DVOPrValR8_sum`) and overseas shares (`DVFShOSVR8_aggr`) are not exported separately; `property_wealth` and the financial wealth totals include them.

Each target draws its quantile with its own seed. microimpute otherwise uses one seed for every target, so a household drawn high for one asset is drawn high for every later target conditioned on it, which makes sparse targets such as the debt secured on land near-certain for the households that hold the asset.

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

5. **Zero non-fuel households**: After imputation, set `petrol_spending` and `diesel_spending` to zero for households where `has_fuel_consumption = 0`

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
