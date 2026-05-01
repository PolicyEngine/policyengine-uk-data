"""Tests for the national OBR council tax compute function.

OBR EFO Table 4.1 reports "Total net council tax receipts" — net of
council tax reduction (CTR). The matching household-level signal is
``council_tax_less_benefit`` (= gross council tax less the CTR
award), not ``council_tax`` (which is the gross liability).

These tests pin the matrix column to the net variable so a future
edit cannot silently regress the gross/net mismatch.
"""

from types import SimpleNamespace

import numpy as np


def _dummy_ctx(council_tax_less_benefit, country):
    """Return a context object compatible with compute_obr_council_tax."""

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.country = np.array(country)

    def pe(variable):
        if variable == "council_tax_less_benefit":
            return np.array(council_tax_less_benefit)
        raise AssertionError(f"unexpected pe call: {variable}")

    ctx.pe = pe
    return ctx


def test_compute_uses_net_variable_not_gross():
    """matrix col for obr/council_tax must be council_tax_less_benefit
    so it matches the OBR net target value. Calibrating gross against
    a net target systematically pushes weights down to fit."""
    from policyengine_uk_data.targets.compute.council_tax import (
        compute_obr_council_tax,
    )

    ctx = _dummy_ctx(
        council_tax_less_benefit=[1000.0, 1500.0, 0.0, 800.0],
        country=["ENGLAND", "ENGLAND", "SCOTLAND", "WALES"],
    )

    out = compute_obr_council_tax(SimpleNamespace(name="obr/council_tax"), ctx)
    np.testing.assert_array_equal(out, [1000.0, 1500.0, 0.0, 800.0])


def test_compute_country_masks_apply_after_net_extraction():
    """Country variants must zero out non-matching households on the
    net variable, not on a gross variable."""
    from policyengine_uk_data.targets.compute.council_tax import (
        compute_obr_council_tax,
    )

    ctx = _dummy_ctx(
        council_tax_less_benefit=[1000.0, 1500.0, 600.0, 800.0],
        country=["ENGLAND", "ENGLAND", "SCOTLAND", "WALES"],
    )

    eng = compute_obr_council_tax(SimpleNamespace(name="obr/council_tax_england"), ctx)
    sco = compute_obr_council_tax(SimpleNamespace(name="obr/council_tax_scotland"), ctx)
    wal = compute_obr_council_tax(SimpleNamespace(name="obr/council_tax_wales"), ctx)

    np.testing.assert_array_equal(eng, [1000.0, 1500.0, 0.0, 0.0])
    np.testing.assert_array_equal(sco, [0.0, 0.0, 600.0, 0.0])
    np.testing.assert_array_equal(wal, [0.0, 0.0, 0.0, 800.0])


def test_compute_does_not_call_gross_variable():
    """If a future refactor reintroduces ctx.pe('council_tax'), the
    DummyCtx will raise AssertionError. Pins the net-only contract."""
    from policyengine_uk_data.targets.compute.council_tax import (
        compute_obr_council_tax,
    )

    ctx = _dummy_ctx(
        council_tax_less_benefit=[100.0, 200.0],
        country=["ENGLAND", "WALES"],
    )

    # No exception means only council_tax_less_benefit was queried.
    compute_obr_council_tax(SimpleNamespace(name="obr/council_tax"), ctx)
