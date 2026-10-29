"""Reduce licensed microdata to counts without letting a failure print a record.

A data test that fails can print the values it checks: pytest shows the
operands of a failing assert, ``-l`` shows each frame's locals, and pandas and
numpy quote the offending value in some error messages ("could not convert
string to float: ..."). Tests on the FRS reduce a dataset to counts through
``counts_or_fail`` and assert only on what it returns.
"""

import pytest


def counts_or_fail(count, dataset, *args):
    """``count(dataset, *args)``, or a test failure that quotes no record.

    An error inside ``count`` becomes a failure that names only ``count`` and
    the error's type, and the error's own traceback goes with it. The function
    raises the failure after the ``except`` block, so it chains no exception.
    ``pytrace=False`` suppresses pytest's usual traceback; ``--full-trace``
    overrides that and shows the frames down to this function, whose only
    locals are ``count``, ``dataset``, ``args`` and the type's name.
    Warnings that ``-W error`` turns into exceptions take the same path.
    ``dataset`` and ``args`` must not show records in their reprs (a
    UKSingleYearDataset shows only its class and address).
    """
    try:
        return count(dataset, *args)
    except Exception as error:
        error_type = type(error).__name__
    raise pytest.fail.Exception(
        f"{count.__name__} raised {error_type}; its message is withheld "
        "because it can quote a record",
        pytrace=False,
    )
