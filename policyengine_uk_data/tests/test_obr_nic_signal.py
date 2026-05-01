"""Regression tests for OBR NIC calibration signal.

Issue #378 (closing) and partial close of #88: every active OBR NIC
target must produce a non-trivial calibration matrix column.

Active targets (against the March 2026 EFO format):

- ``obr/ni_employee`` — Class 1 employee, formula-derived in PE-UK.
- ``obr/ni_employer`` — Class 1 employer, formula-derived in PE-UK.
- ``obr/ni_self_employed`` — combined Class 2 + Class 4, aligned to the
  PE-UK ``ni_self_employed`` variable.

Class 3 is intentionally absent because no dataset populates
``ni_class_3`` — the matrix column would be a flat zero.

Two layers:

1. Registry layer (no Microsimulation) — assert the three expected NIC
   targets are present in the OBR target set and Class 3 is absent.
2. Signal layer (gated on the enhanced FRS fixture) — assert each
   active NIC variable produces non-zero variation across households,
   and that Class 3 would not. Catches future regressions where someone
   re-adds the target without an accompanying imputation.
"""

from __future__ import annotations

import pytest


_ACTIVE_TOPLINE_TARGET_NAMES = (
    "obr/ni_employee",
    "obr/ni_employer",
    "obr/ni_self_employed",
)
_PE_UK_NIC_VARIABLES_WITH_SIGNAL = (
    "ni_employee",
    "ni_employer",
    "ni_self_employed",
    "ni_class_2",
    "ni_class_4",
)


# ── Layer 1: registry contract ──────────────────────────────────────


def test_obr_nic_target_registry_includes_active_classes():
    """The OBR target source must emit the three top-line NIC class targets."""
    from policyengine_uk_data.targets import get_all_targets

    expected = set(_ACTIVE_TOPLINE_TARGET_NAMES)
    actual = {t.name for t in get_all_targets() if t.name in expected}
    assert actual == expected, f"Missing OBR NIC targets: {expected - actual}"


def test_obr_ni_self_employed_target_uses_direct_pe_variable():
    """The combined self-employed target must map directly to the PE-UK
    ``ni_self_employed`` variable so target lineage stays explicit."""
    from policyengine_uk_data.targets import get_all_targets

    target = next(
        (t for t in get_all_targets() if t.name == "obr/ni_self_employed"),
        None,
    )
    assert target is not None, "obr/ni_self_employed not registered"
    assert target.variable == "ni_self_employed"
    assert target.custom_compute is None


def test_obr_ni_class_3_target_is_intentionally_absent():
    """Class 3 must not appear in the registered targets (#378)."""
    from policyengine_uk_data.targets import get_all_targets

    obr_targets = [t for t in get_all_targets() if t.source == "obr"]
    assert "obr/ni_class_3" not in {t.name for t in obr_targets}
    assert "ni_class_3" not in {t.variable for t in obr_targets}


# ── Layer 2: simulator signal ───────────────────────────────────────


@pytest.mark.parametrize("variable", _PE_UK_NIC_VARIABLES_WITH_SIGNAL)
def test_active_nic_variable_has_nonzero_variation(enhanced_frs, variable):
    """Each active NIC variable must produce variation across households,
    otherwise the calibration matrix column is a flat constant and the
    optimiser cannot match its target."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = enhanced_frs.time_period
    values = sim.calculate(variable).values

    nonzero = int((values != 0).sum())
    assert nonzero > 0, f"{variable}: all values are zero — calibration would be inert"
    assert float(values.var()) > 0.0, (
        f"{variable}: zero variance — calibration cannot move it"
    )


def test_ni_class_3_simulator_returns_uniform_zero(enhanced_frs):
    """Direct evidence for why Class 3 is excluded: the simulator produces
    a flat-zero vector, so any calibration target on it is inert. If this
    ever stops being true (e.g. policyengine-uk adds a formula or this
    repo adds an imputation), the Class 3 target should be re-enabled in
    obr.py and the corresponding skip removed."""
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=enhanced_frs)
    sim.default_calculation_period = enhanced_frs.time_period
    values = sim.calculate("ni_class_3").values

    assert (values == 0).all(), (
        f"ni_class_3 has {(values != 0).sum()} non-zero entries — "
        "if intended, restore the target in obr.py and update this test."
    )
