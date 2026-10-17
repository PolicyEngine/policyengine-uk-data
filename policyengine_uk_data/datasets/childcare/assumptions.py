"""Fixed modelling assumptions for the childcare programmes.

Deliberately free of imports: `frs.py` builds the dataset and must not pull
SciPy, `Microsimulation` or Hugging Face configuration in through the
calibration module to read two constants.

**Extended childcare hours.** The distribution of
``maximum_extended_childcare_hours_usage``, clipped to 0-30 hours: the most
funded hours a week a family uses for each child once eligible for the
working parent entitlement. Since policyengine-uk#2177 it caps the child's
total funded hours, with any universal or targeted hours as a floor.

The mean and sd are set from DfE's registered hours, not fitted by the
calibration: the objective sees the distribution only through whether a
benefit unit's clipped draw is positive, which depends on mu/sigma alone, so
it cannot identify them. DfE, "Funded early education and childcare",
reporting year 2026, gives the average weekly registered hours per child
(based on 38 weeks) for children aged 9 months to 2 years on the working
parent entitlement:

    January 2025, entitlement 15 hours a week:  14.6
    January 2026, entitlement 30 hours a week:  26.9

https://explore-education-statistics.service.gov.uk/find-statistics/funded-early-education-and-childcare/2026

For X ~ Normal(mu, sd) clipped to 0-30, E[min(X, 15)] = 14.6 and
E[min(X, 30)] = 26.9 give mu = 36.026, sd = 14.111 (to within 0.002 hours;
test_childcare_targets.py checks both). The same distribution puts 93% of
families above 15 hours, close to DfE's 91% of eligible 3 and 4-year-olds
registered for the working parent entitlement in January 2026, which the
model reads as hours beyond the universal 15. DfE publishes no average for
3 and 4-year-olds, so their hours are assumed to follow the same
distribution.

The previous values, mean 15.019 and sd 4.972, were fitted when the extended
programme carried an untraceable spending target and then held fixed. They
put the average at 15 hours a week, half the January 2026 figure.
"""

EXTENDED_HOURS_MEAN = 36.026
EXTENDED_HOURS_SD = 14.111
