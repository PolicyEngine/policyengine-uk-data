from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from pydantic import BaseModel

FRS_TABLE_NAMES: Tuple[str, ...] = (
    "adult",
    "child",
    "accounts",
    "benefits",
    "job",
    "oddjob",
    "benunit",
    "househol",
    "chldcare",
    "pension",
    "maint",
    "mortgage",
    "penprov",
)


class FRS(BaseModel):
    adult: pd.DataFrame
    child: pd.DataFrame
    accounts: pd.DataFrame
    benefits: pd.DataFrame
    job: pd.DataFrame
    oddjob: pd.DataFrame
    benunit: pd.DataFrame
    househol: pd.DataFrame
    chldcare: pd.DataFrame
    pension: pd.DataFrame
    maint: pd.DataFrame
    mortgage: pd.DataFrame
    penprov: pd.DataFrame

    model_config = {"arbitrary_types_allowed": True}


def load_frs_tables(
    ukda_tab_folder: Union[str, Path],
) -> FRS:
    tables: Dict[str, pd.DataFrame] = {}

    for table_name in FRS_TABLE_NAMES:
        tables[table_name] = pd.read_csv(
            Path(ukda_tab_folder) / f"{table_name}.tab",
            low_memory=False,
            delimiter="\t",
        ).apply(pd.to_numeric, errors="coerce")
        tables[table_name].columns = tables[table_name].columns.str.upper()

    return FRS(
        adult=tables["adult"],
        child=tables["child"],
        accounts=tables["accounts"],
        benefits=tables["benefits"],
        job=tables["job"],
        oddjob=tables["oddjob"],
        benunit=tables["benunit"],
        househol=tables["househol"],
        chldcare=tables["chldcare"],
        pension=tables["pension"],
        maint=tables["maint"],
        mortgage=tables["mortgage"],
        penprov=tables["penprov"],
    )
