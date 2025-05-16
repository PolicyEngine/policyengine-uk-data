from policyengine_core.data import Dataset
from policyengine_uk_data.utils.imputations import *
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.datasets.frs.extended_frs import ExtendedFRS_2022_23
from policyengine_uk_data.datasets.frs.frs import FRS_2022_23
from policyengine_uk_data.utils.loss import create_target_matrix

from policyengine_uk_data.utils.imputations.capital_gains import (
    impute_cg_to_dataset,
)
from policyengine_uk_data.utils.reweight import reweight

try:
    import torch
    from policyengine_uk_data.utils.reweight import reweight
except ImportError:
    torch = None


class EnhancedFRS(Dataset):
    def generate(self):
        data = self.input_frs(require=True).load_dataset()
        self.save_dataset(data)

        # Capital gains imputation

        impute_cg_to_dataset(self)
        data = self.load_dataset()

        self.add_random_variables(data)
        self.add_inferred_disability(data)

        data = self.load_dataset()

        self.save_dataset(data)

    def add_random_variables(self, data: dict):
        from policyengine_uk import Microsimulation

        simulation = Microsimulation(dataset=self)
        RANDOM_VARIABLES = [
            "would_evade_tv_licence_fee",
            "would_claim_pc",
            "would_claim_uc",
            "would_claim_child_benefit",
            "main_residential_property_purchased_is_first_home",
            "household_owns_tv",
            "is_higher_earner",
            "attends_private_school",
        ]
        INPUT_PERIODS = list(range(self.time_period, self.time_period + 10))
        for variable in RANDOM_VARIABLES:
            simulation.get_holder(variable).delete_arrays()
        for variable in RANDOM_VARIABLES:
            value = simulation.calculate(variable, self.time_period).values
            data[variable] = {period: value for period in INPUT_PERIODS}

        self.save_dataset(data)

    def add_inferred_disability(self, data: dict):
        from policyengine_uk import Microsimulation

        simulation = Microsimulation(dataset=self)
        person = simulation.populations["person"]
        parameters = simulation.tax_benefit_system.parameters

        INPUT_PERIODS = list(range(self.time_period, self.time_period + 10))
        WEEKS_IN_YEAR = 52
        THRESHOLD_SAFETY_GAP = 10 * WEEKS_IN_YEAR
        data["is_disabled_for_benefits"] = {}
        data["is_enhanced_disabled_for_benefits"] = {}
        data["is_severely_disabled_for_benefits"] = {}
        for period in INPUT_PERIODS:
            benefit = parameters(period).gov.dwp
            data["is_disabled_for_benefits"][period] = (
                person("dla", period) + person("pip", period) > 0
            )
            data["is_enhanced_disabled_for_benefits"][period] = (
                person("dla_sc", period)
                > benefit.dla.self_care.higher * WEEKS_IN_YEAR
                - THRESHOLD_SAFETY_GAP
            )
            # Child Tax Credit Regulations 2002 s. 8
            paragraph_3 = (
                person("dla_sc", period)
                >= benefit.dla.self_care.higher * WEEKS_IN_YEAR
                - THRESHOLD_SAFETY_GAP
            )
            paragraph_4 = (
                person("pip_dl", period)
                >= benefit.pip.daily_living.enhanced * WEEKS_IN_YEAR
                - THRESHOLD_SAFETY_GAP
            )
            paragraph_5 = person("afcs", period) > 0
            data["is_severely_disabled_for_benefits"][period] = (
                sum([paragraph_3, paragraph_4, paragraph_5]) > 0
            )

        extended_would_claim = (
            np.random.random(len(simulation.calculate("benunit_id"))) < 0.812
        )
        tfc_would_claim = (
            np.random.random(len(simulation.calculate("benunit_id"))) < 0.586
        )
        universal_would_claim = (
            np.random.random(len(simulation.calculate("benunit_id"))) < 0.563
        )
        targeted_would_claim = (
            np.random.random(len(simulation.calculate("benunit_id"))) < 0.597
        )

        # Generate extended childcare hours usage values with mean 15.019 and sd 4.972
        benunit_count = len(simulation.calculate("benunit_id"))
        extended_hours_values = np.random.normal(15.019, 4.972, benunit_count)
        # Clip values to be between 0 and 30 hours
        extended_hours_values = np.clip(extended_hours_values, 0, 30)

        data["would_claim_extended_childcare"] = {
            period: extended_would_claim for period in INPUT_PERIODS
        }
        data["would_claim_tfc"] = {
            period: tfc_would_claim for period in INPUT_PERIODS
        }
        data["would_claim_universal_childcare"] = {
            period: universal_would_claim for period in INPUT_PERIODS
        }
        data["would_claim_targeted_childcare"] = {
            period: targeted_would_claim for period in INPUT_PERIODS
        }

        # Add the maximum extended childcare hours usage
        data["maximum_extended_childcare_hours_usage"] = {
            period: extended_hours_values for period in INPUT_PERIODS
        }

        self.save_dataset(data)


class ReweightedFRS_2022_23(EnhancedFRS):
    name = "reweighted_frs_2022_23"
    label = "Reweighted FRS (2022-23)"
    file_path = STORAGE_FOLDER / "reweighted_frs_2022_23.h5"
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_frs = FRS_2022_23
    time_period = 2022
    end_year = 2022


class EnhancedFRS_2022_23(EnhancedFRS):
    name = "enhanced_frs_2022_23"
    label = "Enhanced FRS (2022-23)"
    file_path = STORAGE_FOLDER / "enhanced_frs_2022_23.h5"
    data_format = Dataset.TIME_PERIOD_ARRAYS
    input_frs = ExtendedFRS_2022_23
    time_period = 2022
    end_year = 2028


if __name__ == "__main__":
    ReweightedFRS_2022_23().generate()
    EnhancedFRS_2022_23().generate()
