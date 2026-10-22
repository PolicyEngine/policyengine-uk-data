# Property wealth: required companion model changes

This document specifies the `policyengine-uk` changes required before building
a data release with all seven WAS property terms mapped to six saved household
components. [policyengine-uk-data#543](https://github.com/PolicyEngine/policyengine-uk-data/pull/543)
is stacked on [#501](https://github.com/PolicyEngine/policyengine-uk-data/pull/501)
and must merge after it. Its base tracks the `was-wealth-split` branch and must
be restacked whenever that branch moves, without rewriting #501's history.
The data PR remains a draft for a batched release. No model PR is opened by
this work.

The property companion must stack on #501's two open model companions:
[policyengine-uk#1974 (`wealth-share-components`)](https://github.com/PolicyEngine/policyengine-uk/pull/1974)
and [#1969 (`uc-capital-sale-expenses`)](https://github.com/PolicyEngine/policyengine-uk/pull/1969).
Retain their share-component, secured-debt and capital-valuation treatment.
Historical file:line citations below describe the audited `policyengine-uk
2.93.0` wheel; they do not describe the current companion branches. Data
citations marked "original PR" refer to head
`af64d3757c2648e820219faee96186d3d7cd9a63`. Current companion behavior below
was checked against the live branches on 13 October 2026.

## Source definition and routing

The [round-8 derived-variable specification](https://doc.ukdataservice.ac.uk/doc/7215/mrdoc/excel/7215_property_dv_spec_r8.xlsx),
worksheet `DV_Spec_Property_R8`, row 49, defines:

```text
DVPropertyR8 = DVhValueR8 + DVHseValR8_sum + DVBltValR8_sum
             + DVBlDValR8_sum + DVLUKValR8_sum + DVLOSValR8_sum
             + DVOprValR8_sum
```

This is gross property value. Row 45 defines the separate net measure
`HpropWR8 = DVPropertyR8 - HmortGR8`. The [user guide, p. 39](https://doc.ukdataservice.ac.uk/doc/7215/mrdoc/pdf/7215_was_userguide_round_8.pdf)
describes mortgage and equity-release deductions from net property wealth.

| WAS field | Stored household component | Coverage |
|---|---|---|
| `DVhValueR8` | `main_residence_value` | Main residence |
| `DVHseValR8_sum` | `other_residential_property_value` | Second UK houses, including time-shares and holiday homes; combined with buy-to-let by #501 |
| `DVBltValR8_sum` | `other_residential_property_value` | UK residential property let for profit; combined with second homes by #501 |
| `DVBlDValR8_sum` | `non_residential_property_value` | Other UK buildings |
| `DVLUKValR8_sum` | `owned_land` | UK land-only holdings |
| `DVLOSValR8_sum` | `overseas_property_value` | Overseas land and property, including time-shares |
| `DVOprValR8_sum` | `other_property_value` | Other real estate; geography and property type are not separately identified |

#501 renames the buy-to-let source to the internal `buy_to_let_value` and
combines it with `second_homes_value` in `other_residential_property_value`.
It also combines their debt in `other_residential_property_secured_debt`.
There is no separately exported buy-to-let model input: adding one to an
aggregate or capital list would count the same holding twice.

Rows 54–59 identify the six other-property categories. The [round-8 dictionaries](https://doc.ukdataservice.ac.uk/doc/7215/mrdoc/ukda_data_dictionaries.zip),
person dictionary `UProp2R8` (position 3,709), identify category 5 as overseas
land **or property**, including time-shares. The [questionnaire](https://doc.ukdataservice.ac.uk/doc/7215/mrdoc/pdf/7215_was_questionnaire_round_8_covid19_version.pdf),
pp. 197–199 and 205–206, confirms the category routing and gross valuations.
The abbreviated `DVLOSValR8_sum` label "overseas land" must not be interpreted
as excluding overseas dwellings.

At the original PR baseline, `DVLUKValR8_sum` already mapped directly to
`owned_land` (`datasets/imputations/wealth.py:73`), and the three building
components mapped at `:102–104`. Overseas assets were renamed `non_uk_land`
at `:97` but omitted from `IMPUTE_VARIABLES` at `:48–60`; buy-to-let and other
real estate had no mapping. The QRF trained on those output columns at
`:318–320`, predicted at `:370`, and wrote each non-conditioning household
output at `:288–297`. Thus `owned_land` means the WAS UK land component,
not estimated land under dwellings.

The restacked mapping retains all seven source terms in six saved components.
The source total remains a sequential conditioning output and is not saved;
an inherited saved `property_wealth` column must also be removed. The QRF
wrapper passes configured targets to MicroImpute (`utils/qrf.py:52–56` at
the original baseline) and delegates prediction (`:68`). Sequential draws
do not impose an accounting constraint tying component draws to the earlier
conditioning-total draw. Source reconciliation is performed before prediction;
the model subsequently derives property wealth from the saved components.

Before training, handle documented `-9` (not applicable), `-8` (don't know or
refusal), and missing property values consistently; report those counts in
source reconciliation. Household dictionary positions 828–829 explicitly
document these codes for overseas and other property. Complete-case and
operational zero-filled reconciliation have different interpretations and
must be reported separately. Do not infer a source residual from independently
imputed release columns or apply the regional HPI adjustment before measuring
the source identity.

## Consumer audit

### Historical 2.93.0 audit

Paths in this table are relative to `policyengine_uk/` in the model repository.
These line numbers and behaviors belong to the historical audited wheel.

| Consumer | File:line in 2.93.0 | Historical behavior |
|---|---|---|
| Residential property | `variables/household/wealth/residential_property_value.py:13` | Adds main residence and the other-residential component, which then contained second homes only |
| Property wealth | `variables/household/wealth/property_wealth.py:12` | Adds residential and non-residential property |
| Direct land | `variables/input/owned_land.py:7,12` | Land-only plots; GDP-per-capita uprating |
| Household land | `variables/household/wealth/household_land_value.py:17–18,56` | Multiplies property wealth by regional intensity, then adds direct land |
| Total wealth | `variables/household/wealth/total_wealth.py:12–17` | Includes property wealth but does not separately add `owned_land` |
| Wealth deciles | `variables/household/wealth/household_wealth_decile.py:12–16` | Reads total wealth with person-weighted household weights |
| Wealth tax | `variables/contrib/ubi_center/wealth_tax.py:13–15` | Reads total wealth |
| Non-primary-residence wealth tax | `variables/contrib/cec/non_primary_residence_wealth_tax.py:13–16` | Reads total wealth minus main residence |
| LVT | `variables/contrib/ubi_center/land_value_tax.py:13–16` | Reads household, corporate and combined land estimates |
| Main dwelling purchases | `variables/household/consumption/main_residential_property_purchased.py:14–15` | Reads main residence directly |
| Additional dwelling purchases | `variables/household/consumption/additional_residential_property_purchased.py:14–17` | Reads the other-residential component directly |
| Non-residential purchases | `variables/household/consumption/non_residential_property_purchased.py:12` | Reads the non-residential component directly |
| High-value council tax | `variables/gov/hmrc/high_value_council_tax_surcharge.py:23` | Reads main residence directly |
| UC capital | `variables/gov/dwp/universal_credit/uc_assessable_capital.py:19–21` | Sums configured capital sources |
| UC property-income gate | `variables/gov/dwp/universal_credit/income/uc_unearned_income.py:18–20,26–27` | Tested other-residential and non-residential capital; this gate has since been removed |
| Housing Benefit capital | `variables/gov/dwp/housing_benefit/housing_benefit_assessable_capital.py:20–21` | Sums configured capital sources |
| Pension Credit capital | `variables/gov/dwp/pension_credit/pension_credit_assessable_capital.py:21–22` | Sums configured capital sources |

### Verified current companion behavior

#1974 changes all six countable-capital lists to use `share_like_capital`
instead of `corporate_wealth`, retaining the share-component aggregate
fallback and avoiding duplicate holdings. #1969 uses
[`utils/capital_valuation.py`](https://github.com/PolicyEngine/policyengine-uk/blob/uc-capital-sale-expenses/policyengine_uk/utils/capital_valuation.py)
to apply each programme's source-specific sale-expense deduction and then
secured debt, flooring secured-property categories at zero. Its debt mapping
links `other_residential_property_value` to
`other_residential_property_secured_debt`; that single link covers second
homes and buy-to-let together after #501. Preserve the combined stack's share
routing when updating property sources; do not restore historical capital
lists or overwrite share sale-expense decisions.

The current
[`uc_unearned_income.py`](https://github.com/PolicyEngine/policyengine-uk/blob/uc-capital-sale-expenses/policyengine_uk/variables/gov/dwp/universal_credit/income/uc_unearned_income.py)
sums configured claimant unearned-income sources. Actual interest, dividends
and rent are excluded at every capital level; capital enters through tariff
income. There is no property-income gate to extend for the new inputs.

The current
[`additional_residential_property_purchased.py`](https://github.com/PolicyEngine/policyengine-uk/blob/uc-capital-sale-expenses/policyengine_uk/variables/household/consumption/additional_residential_property_purchased.py)
multiplies `other_residential_property_value` by `property_purchased`. #501
therefore includes domestic buy-to-let automatically once in the existing
additional-dwelling purchase/SDLT stock proxy. No separate buy-to-let term is
needed. Overseas and unclassified other real estate remain outside that proxy.

Income Support, income-based JSA and income-related ESA likewise use the
configured capital-source lists specified below. Benefit capital,
property-purchase proxies and high-value council tax do not read
`property_wealth` directly. Removing the stored aggregate alone has no direct
effect on these calculations at fixed weights; completing the component
mapping and companion routing can change them.

In the original uk-data audit, `targets/build_loss_matrix.py:305–315`
dispatches existing national and regional land targets. No control values,
target weights or objective terms change. Calculated land columns, evaluated
losses and fitted constituency/local-authority weights can change.
Recalibration can change weighted outputs even where formulas have no direct
aggregate dependency.

## Exact model implementation

### New stock inputs and uprating

Create only `policyengine_uk/variables/input/overseas_property_value.py` and
`other_property_value.py`. Each follows this template, substituting its class
name, label and category documentation:

```python
from policyengine_uk.model_api import *


class overseas_property_value(Variable):
    label = "Overseas property value"
    documentation = "Gross value of overseas land and property, including time-shares"
    entity = Household
    definition_period = YEAR
    value_type = float
    unit = GBP
    quantity_type = STOCK
    default_value = 0
    uprating = "gov.economic_assumptions.indices.obr.per_capita.gdp"
```

Describe other property as gross other real estate with unspecified
geography/type. This GDP index matches existing main residence,
other-residential, non-residential and land input attributes. Add both new
names under `gov.economic_assumptions.yoy_growth.obr.per_capita.gdp` in
`policyengine_uk/data/uprating_indices.yaml`, preserving #501's model
companions' registrations. Preserve stock handling for monthly calculations.

Both registrations matter. In the historical loading audit, single-year
dataset loading copied columns into each year (`simulation.py:468–470` and
`data/economic_assumptions.py:43–51`) and grew only names in the YAML list
(`data/economic_assumptions.py:89–98`). Those explicit annual inputs override
the generic core fallback using a variable's `uprating` attribute when only
an earlier-period input exists. A companion test must cover the dataset-load
route as well as situation-based calculations. Historical dataset `validate()`
checked missing values only (`data/dataset_schema.py:145–150`); the simulation
loader skipped unknown columns (`simulation.py:499–505`). The uk-data
compatibility guard must prevent release generation with a model that cannot
carry the new components.

### Aggregates and land

Keep `variables/household/wealth/residential_property_value.py` with its two
existing terms:

```python
adds = [
    "main_residence_value",
    "other_residential_property_value",
]
```

Document domestic residential property: the second term now includes both
second homes and buy-to-let through #501. Overseas and other real estate
cannot be classified as residential from their aggregate WAS values.

In `variables/household/wealth/property_wealth.py`:

```python
adds = [
    "residential_property_value",
    "non_residential_property_value",
    "owned_land",
    "overseas_property_value",
    "other_property_value",
]
```

This gives:

```text
property_wealth = residential_property_value + non_residential_property_value
                + owned_land + overseas_property_value + other_property_value
```

Document worldwide gross property value, including direct UK land holdings.
`total_wealth` receives these assets through `property_wealth`; do not also
add land, overseas property or other property separately to `total_wealth`.
Gross asset wealth remains separate from #1969's net countable-capital
valuation, which deducts sale expenses and identified secured debt.

In `variables/household/wealth/household_land_value.py`, keep the existing
region selection and intensity parameters, add the overseas input and replace
the return expression:

```python
overseas_property_value = household("overseas_property_value", period)
return (
    property_wealth - owned_land - overseas_property_value
) * property_wealth_intensity + owned_land
```

Thus `household_land_value = (property_wealth - owned_land -
overseas_property_value) * regional_intensity + owned_land`. Direct UK land
enters once at full value; explicitly identified overseas holdings enter zero
times. The remaining other-real-estate value uses the existing regional
intensity approximation, subject to the geography question below.

### Benefit capital and domestic purchase routing

Add `overseas_property_value` and `other_property_value` exactly once to each
countable-capital `sources.yaml` and its `sale_expenses/sources.yaml` below.
Preserve other sources, effective dates and the combined companions' share
treatment. Update descriptions to identify the new categories and retain
their broad-asset, beneficial-ownership and disregard limitations.

| Programme | Capital directory under `policyengine_uk/parameters/` | Lists to update |
|---|---|---|
| Universal Credit | `gov/dwp/universal_credit/means_test/capital/` | `sources.yaml`, `sale_expenses/sources.yaml` |
| Housing Benefit | `gov/dwp/housing_benefit/means_test/capital/` | `sources.yaml`, `sale_expenses/sources.yaml` |
| Income Support | `gov/dwp/income_support/means_test/capital/` | `sources.yaml`, `sale_expenses/sources.yaml` |
| Income-based JSA | `gov/dwp/JSA/income/capital/` | `sources.yaml`, `sale_expenses/sources.yaml` |
| Income-related ESA | `gov/dwp/ESA/income/capital/` | `sources.yaml`, `sale_expenses/sources.yaml` |
| Pension Credit | `gov/dwp/pension_credit/income/capital/` | `sources.yaml`, `sale_expenses/sources.yaml` |

Both types of list matter: #1969 deducts the programme's sale-expense rate
only when a counted source is also in its sale-expense list. Listing a source
for sale expenses alone does not make it countable capital. Overseas and other
real estate use the same real-property sale-expense treatment as the existing
property categories. Preserve #1969's helper and secured-debt mapping; do not
add a separate buy-to-let source or debt. No additional secured-debt input is
exported for overseas or other real estate, so debt on those holdings cannot
be identified or deducted by this mapping.

The new categories enter non-main-home capital within the model's current
proxy, reported-capital and disregard semantics. Source aggregates cannot
identify additional statutory disregards or individual beneficial ownership.
This is not a claim that every holding is legally countable in every
household. Keep main residence and private pension rights excluded.

UC currently omits `owned_land` from its countable-capital list, while the
other five lists include it. Preserve that existing UC treatment; correcting
it requires the explicit model decision recorded below. Its presence in a
sale-expense list does not override that omission.

Retain the current additional residential purchase formula:

```python
other_residential_property_value = household("other_residential_property_value", period)
return other_residential_property_value * property_purchased
```

The #501 component already includes second homes and domestic buy-to-let
once. Keep overseas assets and unclassified other real estate outside the UK
dwelling-purchase/SDLT proxy. This retains the existing stock-based purchase
approximation; it does not establish observed transaction values.

## Required model tests and invariants

Add `policyengine_uk/tests/test_property_wealth_components.py` with synthetic
single-household tests and Hypothesis tests of non-negative finite component
values. No licensed data is required. The invariants are:

- The seven WAS source terms reconcile before imputation, with second homes
  and buy-to-let summed into one saved other-residential component.
- Gross property wealth equals the sum of the six saved property inputs.
- Buy-to-let contributes exactly once to residential property, gross property
  and total wealth, capital and the domestic additional-dwelling purchase proxy.
- With other wealth inputs zero, total wealth equals that property sum.
- UK land-only value contributes once to property and once to household land.
- Overseas value contributes to global property/total wealth and contributes
  zero to household land and household LVT.
- Uprating each saved input preserves the aggregate identity at every tested
  year; monthly stock values are not prorated.

Cover each of the seven source categories alone and a mixed household. In
particular, verify second-home-only, buy-to-let-only and combined values use
the same other-residential input. In London, £100,000 domestic buildings,
£20,000 direct UK land and £30,000 overseas assets give £150,000 property
wealth and £105,000 household land with intensity 0.85. Repeat at 2024 and
2026 using the GDP index; check each input and aggregate growth against the
actual model index. Test monthly stock values to detect annual-value prorating.

Add/extend `tests/policy/baseline/household/wealth/household_land_value.yaml`
with land-only, overseas-only and mixed cases. Preserve existing regional
cases; for consistent component inputs their expected land values remain
valid. Test household LVT with only its household rate enabled.

Add separate overseas-only and other-property-only cases to each affected
benefit-capital test file, including sale-expense valuation, reported-capital
overrides and existing household-to-benunit allocation behavior. Assert each
new source occurs exactly once in both relevant lists, and main residence
remains excluded. A combined second-home/buy-to-let case must use #501's
single other-residential value and combined secured debt, with #1969's
sale-expense deduction before debt and zero floor. Preserve current UC
unearned-income behavior; no property-income gate test is required.

Test that buy-to-let enters the existing additional domestic dwelling-purchase
proxy once, while overseas and unclassified other holdings do not. Keep the
uk-data stored-total override regression. Demonstrate separate mutations for
omitting a source category, double-counting buy-to-let, restoring saved-total
persistence, dropping the direct-land subtraction and dropping the overseas
subtraction; each must fail its relevant test before restoring source files
byte for byte.

Until the property companion is released, a test-only Reform can exercise
the specified contract. Passing it proves the contract under that Reform,
not implementation by the installed model or production release. For
dataset-based contract tests, arrange for the new definitions to exist before
data loading and mock the YAML registration in the fixture; adding only
`Variable.uprating` to a Reform does not exercise the required dataset
uprating route. Keep test-only support out of production data files.

## Reviewer decisions and release boundary

The proposed treatment keeps overseas and direct UK land in global gross
`property_wealth`, matching WAS and the model's "all owned properties"
semantics, then corrects the domestic land base. Alternatives to consider
are keeping overseas assets only in `total_wealth`, or keeping UK land
outside `property_wealth` and adding it separately to `total_wealth`. Either
requires changing aggregate definitions and consumer expectations.

The other-real-estate category has no explicit geography/type in its public
label. The selected treatment retains it as separate global wealth and uses
the current regional property-to-land approximation for land estimation.
Reviewers should decide whether additional source detail permits geographic
classification, or whether to exclude unclassified holdings from UK land
until that classification exists. The lack of identified secured debt for
overseas and other real estate also limits their benefit-capital valuation.

Reviewers should separately decide whether to add `owned_land` to UC capital
sources; its present omission is inconsistent with the other listed benefits
but is not changed implicitly through the gross property aggregate.

Merge #501 first. Implement and release this property companion on top of
the open #1974 and #1969 model companions, update the data dependency/lock to
a release carrying all required changes, and run its targeted tests before
the batched release. The data guard must reject a model that cannot carry the
new components. Licensed WAS reconciliation remains unrun in this workspace;
the seven-term reconciliation tool must be run on the licensed household
source before release.

Production rebuilding, recalibration and LVT-impact measurement remain
separate release work. Existing data releases are not migrated; their saved
totals still override formulas. Only main-residence imputation receives the
regional HPI adjustment; the remaining source categories retain their mapped
values and the model's chosen uprating index, with second homes and buy-to-let
already combined by #501.
