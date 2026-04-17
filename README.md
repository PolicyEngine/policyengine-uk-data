# PolicyEngine UK Data

PolicyEngine's project to build accurate UK household survey data.

## Local dataset builds

For a full local dataset build:

1. Ensure the private prerequisite folders exist under `policyengine_uk_data/storage/`.
2. Use Python 3.13. Python 3.14 currently fails while loading PyTables/Blosc2 in this repo.
3. Prefer the sibling `policyengine-uk` checkout when building locally, because the published wheel in your active environment may not expose all variables required by the data pipeline.

If `../policyengine-uk` exists, you can run:

```sh
make data-local
```

## Panel ID contract

When the pipeline is extended to produce a sequence of yearly snapshots
(see [issue #345](https://github.com/PolicyEngine/policyengine-uk-data/issues/345)),
three identifier columns are the **panel keys** that link rows across years:

| Table | ID column |
| --- | --- |
| `household` | `household_id` |
| `benunit` | `benunit_id` |
| `person` | `person_id` |

These IDs are deterministic functions of the FRS `sernum` (see
`policyengine_uk_data/datasets/frs.py`) and must be preserved byte-for-byte
by every downstream transformation so that snapshot _Y_ and snapshot _Y + 1_
can be joined on them. Use
`policyengine_uk_data.utils.panel_ids.assert_panel_id_consistency` to check
this invariant when adding new save-time or ageing logic.
