import warnings

import numpy as np
import pytest

from policyengine_uk_data.tests.record_safe import counts_or_fail

SENTINEL = "777.25"


def _negative_values(table, column):
    return int((table[column] < 0).sum())


def _quotes_a_record(table):
    raise ValueError(f"could not convert string to float: '{SENTINEL} pounds'")


def _warns_quoting_a_record(table):
    warnings.warn(f"odd value {SENTINEL}", RuntimeWarning)
    return 0


def test_returns_the_counts():
    table = {"amount": np.array([-1.0, 2.0, -3.0])}
    assert counts_or_fail(_negative_values, table, "amount") == 2


@pytest.mark.parametrize(
    "count, args",
    [
        (_quotes_a_record, ()),
        (_negative_values, ("missing_column",)),
    ],
)
def test_an_error_fails_without_its_message(count, args):
    table = {"amount": np.array([float(SENTINEL)])}
    with pytest.raises(pytest.fail.Exception) as failure:
        counts_or_fail(count, table, *args)
    message = str(failure.value)
    assert message.startswith(f"{count.__name__} raised ")
    assert SENTINEL not in message
    assert failure.value.pytrace is False
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None


def test_a_warning_raised_as_an_error_fails_without_its_message():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(pytest.fail.Exception) as failure:
            counts_or_fail(_warns_quoting_a_record, {})
    message = str(failure.value)
    assert message.startswith("_warns_quoting_a_record raised RuntimeWarning")
    assert SENTINEL not in message
