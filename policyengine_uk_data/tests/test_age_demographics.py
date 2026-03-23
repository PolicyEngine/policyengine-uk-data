"""Tests for country-specific age demographic profiles (#64)."""

import pandas as pd
import numpy as np


def _load_age_csv(path):
    ages = pd.read_csv(path)
    age_cols = [c for c in ages.columns if c not in ["code", "name", "all"]]
    return ages, age_cols


class TestConstituencyAgeDemographics:
    PATH = "policyengine_uk_data/datasets/local_areas/constituencies/targets/age.csv"

    def test_ni_profiles_differ_from_uk_mean(self):
        """NI constituencies should use NI-specific age profiles, not UK mean."""
        ages, age_cols = _load_age_csv(self.PATH)
        ni = ages[ages["code"].str.startswith("N")]
        non_ni = ages[~ages["code"].str.startswith("N")]

        ni_mean = ni[age_cols].mean()
        uk_mean = non_ni[age_cols].mean()

        # NI mean should differ from rest-of-UK mean
        assert not np.allclose(ni_mean.values, uk_mean.values, rtol=0.01), (
            "NI constituency age profiles are still identical to UK mean"
        )

    def test_all_constituencies_have_age_data(self):
        """Every constituency should have non-NaN age data."""
        ages, age_cols = _load_age_csv(self.PATH)
        assert not ages[age_cols].isna().any().any(), (
            "Some constituencies have missing age data"
        )

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
        young_cols = [
            c
            for c in age_cols
            if c.replace("+", "").isdigit() and int(c.replace("+", "")) < 20
        ]

        ni = ages[ages["code"].str.startswith("N")]
        non_ni = ages[~ages["code"].str.startswith("N")]

        ni_young_share = ni[young_cols].sum(axis=1).mean() / ni["all"].mean()
        uk_young_share = non_ni[young_cols].sum(axis=1).mean() / non_ni["all"].mean()

        assert ni_young_share > uk_young_share, (
            f"NI young share ({ni_young_share:.3f}) should exceed "
            f"UK average ({uk_young_share:.3f})"
        )


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
