# Post-CTR Council Tax Calibration

This draft is contingent on UK-wide Council Tax Reduction (CTR) support
landing in PolicyEngine/policyengine-uk#1534. Until then, the data
pipeline should keep using the existing FRS-derived Council Tax inputs
for production outputs.

## Current state

The dataset pipeline starts from FRS Council Tax variables, then
calibrates household weights to public Council Tax targets:

- `council_tax` comes from FRS-reported annual Council Tax, with missing
  values imputed from FRS cells.
- `council_tax_band` comes from the FRS band variable.
- `council_tax_less_benefit` is the PolicyEngine UK signal for net
  Council Tax receipts.
- National OBR Council Tax targets use `council_tax_less_benefit`.
- LA calibration includes Council Tax band counts and net Council Tax
  targets where direct or derived public sources are available.

## Post-CTR target state

After UK-wide CTR coverage is complete, the production pipeline should
be able to validate and then switch to:

```text
structural gross Council Tax liability
- modelled Council Tax Reduction
= net Council Tax paid
```

FRS-reported Council Tax and reported CTB/CTR should remain available as
validation signals, imputation predictors, and fallback diagnostics, but
they should not be the long-run source of truth for policy counterfactual
net Council Tax.

## Acceptance gates before switching

Do not switch production calibration outputs until all of these are true:

- PolicyEngine UK has CTR schemes for every council tax billing
  authority that the dataset can assign.
- Unsupported or unmapped local authority records are explicitly
  counted and small enough to accept, or handled with a documented
  fallback.
- Modelled national and country Council Tax Reduction totals are within
  agreed tolerances of admin spend or caseload targets where available.
- Modelled `council_tax_less_benefit` remains within agreed tolerances of
  OBR net Council Tax receipts.
- LA-level net Council Tax diagnostics are no worse than the current
  FRS-derived calibration baseline for England and Wales where targets
  exist.
- Scotland and Northern Ireland remain explicitly handled rather than
  silently folded into England/Wales assumptions.

## Draft implementation sequence

1. Add compare-only structural gross Council Tax liability outputs from
   PolicyEngine UK.
2. Add diagnostics comparing FRS-reported net Council Tax with structural
   gross liability minus modelled CTR.
3. Add CTR calibration targets and diagnostics for spend, caseload, and
   net receipts where primary public sources support them.
4. Add a guarded pipeline switch that can choose either the current
   FRS-derived net Council Tax signal or the structural/modelled signal.
5. Make the structural/modelled signal the default only after the
   acceptance gates pass in release validation.
