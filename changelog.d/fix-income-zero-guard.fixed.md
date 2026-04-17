Guard the rent/mortgage rescaling in `impute_over_incomes` against `ZeroDivisionError` when the seed dataset's imputation columns sum to zero (e.g. the zero-weight synthetic copy in `impute_income`).
