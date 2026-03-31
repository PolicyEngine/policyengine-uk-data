"""Tests for country-specific age demographic profiles (#64).

Data sources:
- Constituency raw age: House of Commons Library
  https://commonslibrary.parliament.uk/constituency-statistics-population-by-age/
- LA raw age: ONS mid-year population estimates (Nomis)
  https://www.nomisweb.co.uk/datasets/pestsyoala
- NI national age distribution: ONS subnational population projections
  via policyengine_uk_data/storage/demographics.csv
- NI constituency-level (future): NISRA small areas
  https://www.nisra.gov.uk/publications/2024-mid-year-population-estimates-small-geographical-areas-within-northern-ireland
- NI LGD-level (future): NISRA mid-year estimates
  https://www.nisra.gov.uk/publications/2024-mid-year-population-estimates-northern-ireland-and-estimates-population-aged-85
"""

import pandas as pd
import numpy as np


def _load_age_csv(path):
    ages = pd.read_csv(path)
    age_cols = [c for c in ages.columns if c not in ["code", "name", "all"]]
    return ages, age_cols


def _young_cols(age_cols, max_age=20):
    return [
        c
        for c in age_cols
        if c.replace("+", "").isdigit() and int(c.replace("+", "")) < max_age
    ]


class TestConstituencyAgeDemographics:
    PATH = "policyengine_uk_data/datasets/local_areas/constituencies/targets/age.csv"

    def test_ni_profiles_differ_from_uk_mean(self):
        """NI constituencies should use NI-specific age profiles, not UK mean."""
        ages, age_cols = _load_age_csv(self.PATH)
        ni = ages[ages["code"].str.startswith("N")]
        non_ni = ages[~ages["code"].str.startswith("N")]

        ni_mean = ni[age_cols].mean()
        uk_mean = non_ni[age_cols].mean()

        assert not np.allclose(ni_mean.values, uk_mean.values, rtol=0.01), (
            "NI constituency age profiles are still identical to UK mean"
        )

    def test_all_constituencies_have_age_data(self):
        """Every constituency should have non-NaN age data."""
        ages, age_cols = _load_age_csv(self.PATH)
        assert not ages[age_cols].isna().any().any(), (
            "Some constituencies have missing age data"
        )

    def test_expected_constituency_counts(self):
        """Should have 650 constituencies across all four countries."""
        ages, _ = _load_age_csv(self.PATH)
        assert len(ages) == 650
        assert len(ages[ages["code"].str.startswith("E")]) == 533
        assert len(ages[ages["code"].str.startswith("S")]) == 59
        assert len(ages[ages["code"].str.startswith("W")]) == 40
        assert len(ages[ages["code"].str.startswith("N")]) == 18

    def test_scotland_has_distinct_profiles(self):
        """Scotland constituencies should not all be identical."""
        ages, age_cols = _load_age_csv(self.PATH)
        scotland = ages[ages["code"].str.startswith("S")]
        assert len(scotland) > 1
        first = scotland[age_cols].iloc[0]
        assert not all(
            (scotland[age_cols].iloc[i] == first).all() for i in range(1, len(scotland))
        ), "All Scotland constituencies have identical age profiles"

    def test_ni_younger_than_uk_average(self):
        """NI has a younger population — proportion aged 0-19 should be
        higher than the UK average."""
        ages, age_cols = _load_age_csv(self.PATH)
        young = _young_cols(age_cols)

        ni = ages[ages["code"].str.startswith("N")]
        non_ni = ages[~ages["code"].str.startswith("N")]

        ni_young_share = ni[young].sum(axis=1).mean() / ni["all"].mean()
        uk_young_share = non_ni[young].sum(axis=1).mean() / non_ni["all"].mean()

        assert ni_young_share > uk_young_share, (
            f"NI young share ({ni_young_share:.3f}) should exceed "
            f"UK average ({uk_young_share:.3f})"
        )

    def test_age_columns_sum_to_all(self):
        """Sum of single-year age columns should roughly equal the 'all' total."""
        ages, age_cols = _load_age_csv(self.PATH)
        row_sums = ages[age_cols].sum(axis=1)
        assert np.allclose(row_sums, ages["all"], rtol=0.01), (
            "Age columns do not sum to 'all' total"
        )

    def test_no_negative_values(self):
        """Age counts should not be negative."""
        ages, age_cols = _load_age_csv(self.PATH)
        assert (ages[age_cols] >= 0).all().all(), "Negative age counts found"


class TestLocalAuthorityAgeDemographics:
    PATH = "policyengine_uk_data/datasets/local_areas/local_authorities/targets/age.csv"

    def test_ni_profiles_differ_from_uk_mean(self):
        """NI LAs should use NI-specific age profiles, not UK mean."""
        ages, age_cols = _load_age_csv(self.PATH)
        ni = ages[ages["code"].str.startswith("N")]
        non_ni = ages[~ages["code"].str.startswith("N")]

        ni_mean = ni[age_cols].mean()
        uk_mean = non_ni[age_cols].mean()

        assert not np.allclose(ni_mean.values, uk_mean.values, rtol=0.01), (
            "NI LA age profiles are still identical to UK mean"
        )

    def test_all_las_have_age_data(self):
        """Every LA should have non-NaN age data."""
        ages, age_cols = _load_age_csv(self.PATH)
        assert not ages[age_cols].isna().any().any(), "Some LAs have missing age data"

    def test_expected_country_counts(self):
        """Should have LAs from all four countries."""
        ages, _ = _load_age_csv(self.PATH)
        assert len(ages[ages["code"].str.startswith("E")]) > 0, "No England LAs"
        assert len(ages[ages["code"].str.startswith("S")]) > 0, "No Scotland LAs"
        assert len(ages[ages["code"].str.startswith("W")]) > 0, "No Wales LAs"
        assert len(ages[ages["code"].str.startswith("N")]) > 0, "No NI LAs"

    def test_ni_younger_than_uk_average(self):
        """NI LAs should have a higher proportion of young people."""
        ages, age_cols = _load_age_csv(self.PATH)
        young = _young_cols(age_cols)

        ni = ages[ages["code"].str.startswith("N")]
        non_ni = ages[~ages["code"].str.startswith("N")]

        ni_young_share = ni[young].sum(axis=1).mean() / ni["all"].mean()
        uk_young_share = non_ni[young].sum(axis=1).mean() / non_ni["all"].mean()

        assert ni_young_share > uk_young_share

    def test_age_columns_sum_to_all(self):
        """Sum of single-year age columns should roughly equal the 'all' total."""
        ages, age_cols = _load_age_csv(self.PATH)
        row_sums = ages[age_cols].sum(axis=1)
        assert np.allclose(row_sums, ages["all"], rtol=0.01)

    def test_no_negative_values(self):
        """Age counts should not be negative."""
        ages, age_cols = _load_age_csv(self.PATH)
        assert (ages[age_cols] >= 0).all().all(), "Negative age counts found"
