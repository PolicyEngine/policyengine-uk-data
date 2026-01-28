"""Imputation chain assets using original imputation functions."""

from pathlib import Path

from dagster import asset, AssetExecutionContext, Config
from policyengine_uk.data import UKSingleYearDataset
from pydantic import Field

from policyengine_uk_data.resources.bucket import BucketResource


def _load_dataset(data: dict) -> UKSingleYearDataset:
    """Reconstruct dataset from dict."""
    return UKSingleYearDataset(
        person=data["person"],
        benunit=data["benunit"],
        household=data["household"],
        fiscal_year=data.get("fiscal_year", 2023),
    )


def _save_dataset(dataset: UKSingleYearDataset) -> dict:
    """Convert dataset to dict for serialisation."""
    return {
        "person": dataset.person,
        "benunit": dataset.benunit,
        "household": dataset.household,
        "fiscal_year": dataset.time_period,
    }


@asset(group_name="imputations")
def frs_with_wealth(context: AssetExecutionContext, base_frs: dict) -> dict:
    """FRS with wealth variables imputed from WAS."""
    from policyengine_uk_data.datasets.imputations import impute_wealth

    dataset = _load_dataset(base_frs)
    context.log.info("Imputing wealth variables from WAS")
    dataset = impute_wealth(dataset)

    context.add_output_metadata({
        "imputed_columns": [
            "owned_land", "property_wealth", "corporate_wealth",
            "gross_financial_wealth", "net_financial_wealth",
            "main_residence_value", "num_vehicles",
        ],
    })

    return _save_dataset(dataset)


@asset(group_name="imputations")
def frs_with_consumption(
    context: AssetExecutionContext, frs_with_wealth: dict
) -> dict:
    """FRS with consumption variables imputed from LCFS."""
    from policyengine_uk_data.datasets.imputations import impute_consumption

    dataset = _load_dataset(frs_with_wealth)
    context.log.info("Imputing consumption variables from LCFS")
    dataset = impute_consumption(dataset)

    return _save_dataset(dataset)


@asset(group_name="imputations")
def frs_with_vat(
    context: AssetExecutionContext, frs_with_consumption: dict
) -> dict:
    """FRS with VAT expenditure rate imputed from ETB."""
    from policyengine_uk_data.datasets.imputations import impute_vat

    dataset = _load_dataset(frs_with_consumption)
    context.log.info("Imputing VAT variables from ETB")
    dataset = impute_vat(dataset)

    return _save_dataset(dataset)


@asset(group_name="imputations")
def frs_with_services(
    context: AssetExecutionContext, frs_with_vat: dict
) -> dict:
    """FRS with public service usage imputed."""
    from policyengine_uk_data.datasets.imputations import impute_services

    dataset = _load_dataset(frs_with_vat)
    context.log.info("Imputing public service usage")
    dataset = impute_services(dataset)

    return _save_dataset(dataset)


@asset(group_name="imputations")
def frs_with_income(
    context: AssetExecutionContext, frs_with_services: dict
) -> dict:
    """FRS with income variables imputed from SPI."""
    from policyengine_uk_data.datasets.imputations import impute_income

    dataset = _load_dataset(frs_with_services)
    context.log.info("Imputing income variables from SPI")
    dataset = impute_income(dataset)

    return _save_dataset(dataset)


@asset(group_name="imputations")
def frs_with_capital_gains(
    context: AssetExecutionContext, frs_with_income: dict
) -> dict:
    """FRS with capital gains imputed."""
    from policyengine_uk_data.datasets.imputations import impute_capital_gains

    dataset = _load_dataset(frs_with_income)
    context.log.info("Imputing capital gains")
    dataset = impute_capital_gains(dataset)

    return _save_dataset(dataset)


@asset(group_name="imputations")
def frs_with_salary_sacrifice(
    context: AssetExecutionContext, frs_with_capital_gains: dict
) -> dict:
    """FRS with salary sacrifice imputed."""
    from policyengine_uk_data.datasets.imputations import impute_salary_sacrifice

    dataset = _load_dataset(frs_with_capital_gains)
    context.log.info("Imputing salary sacrifice")
    dataset = impute_salary_sacrifice(dataset)

    return _save_dataset(dataset)


class StudentLoanConfig(Config):
    year: int = Field(default=2023, description="Year for student loan plan")


@asset(group_name="imputations")
def frs_with_student_loans(
    context: AssetExecutionContext,
    config: StudentLoanConfig,
    frs_with_salary_sacrifice: dict,
) -> dict:
    """FRS with student loan plan imputed."""
    from policyengine_uk_data.datasets.imputations import impute_student_loan_plan

    dataset = _load_dataset(frs_with_salary_sacrifice)
    context.log.info("Imputing student loan plans")
    dataset = impute_student_loan_plan(dataset, year=config.year)

    return _save_dataset(dataset)


class UpratingConfig(Config):
    target_year: int = Field(default=2025, description="Year to uprate to")


@asset(group_name="imputations")
def uprated_frs(
    context: AssetExecutionContext,
    config: UpratingConfig,
    frs_with_student_loans: dict,
) -> dict:
    """FRS uprated to target year for calibration."""
    from policyengine_uk_data.utils.uprating import uprate_dataset

    dataset = _load_dataset(frs_with_student_loans)
    context.log.info(f"Uprating to {config.target_year}")
    dataset = uprate_dataset(dataset, config.target_year)

    context.add_output_metadata({
        "source_year": frs_with_student_loans.get("fiscal_year", 2023),
        "target_year": config.target_year,
    })

    return _save_dataset(dataset)
