"""Fixed modelling assumptions for the childcare programmes.

Deliberately free of imports: `frs.py` builds the dataset and must not pull
SciPy, `Microsimulation` or Hugging Face configuration in through the
calibration module to read two constants.

**Extended childcare hours.** The distribution of
``maximum_extended_childcare_hours_usage``, clipped to 0-30 hours.

These were fitted when the extended programme still carried a spending
target. Without it, the calibration objective sees the distribution only
through whether a benefit unit's clipped draw is positive, and
``clip(mu + sigma*z, 0, 30) > 0`` depends on mu/sigma alone: (15, 5) and
(30, 10) produce the same mask and so the same loss. Fitting them was
underidentified, so they are held fixed and consumed by both draw sites —
`childcare/takeup_rate.py` and `frs.py`.

Replacing them needs published evidence on the funded hours actually taken,
not a re-run of the optimisation.
"""

EXTENDED_HOURS_MEAN = 15.019
EXTENDED_HOURS_SD = 4.972
