from policyengine_core.data import Dataset
from policyengine_uk_data.utils.imputations import *
from policyengine_uk_data.storage import STORAGE_FOLDER
from policyengine_uk_data.datasets.frs.extended_frs import ExtendedFRS_2022_23
from policyengine_uk_data.datasets.frs.frs import FRS_2022_23
from policyengine_uk_data.utils.loss import create_target_matrix

from policyengine_uk_data.utils.imputations.capital_gains import (
    impute_cg_to_dataset,
)

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

        self.save_dataset(data)

        self.add_random_variables(data)

        # Reweighting

        data = self.load_dataset()
        original_weights = data["household_weight"][str(self.time_period)] + 10
        for year in range(self.time_period, self.end_year + 1):
            loss_matrix, targets_array = create_target_matrix(self, year)
            new_weights = reweight(
                original_weights, loss_matrix, targets_array
            )
            data["household_weight"][str(year)] = new_weights

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
    url = "hf://policyengine/policyengine-uk-data"


def reweight(
    original_weights,
    loss_matrix,
    targets_array,
    dropout_rate=0.05,
):
    target_names = np.array(loss_matrix.columns)
    loss_matrix = torch.tensor(loss_matrix.values, dtype=torch.float32)
    targets_array = torch.tensor(targets_array, dtype=torch.float32)
    weights = torch.tensor(
        np.log(original_weights), requires_grad=True, dtype=torch.float32
    )

    # TODO: replace this with a call to the python reweight.py package.
    def loss(weights):
        # Check for Nans in either the weights or the loss matrix
        if torch.isnan(weights).any():
            raise ValueError("Weights contain NaNs")
        if torch.isnan(loss_matrix).any():
            raise ValueError("Loss matrix contains NaNs")
        estimate = weights @ loss_matrix
        if torch.isnan(estimate).any():
            raise ValueError("Estimate contains NaNs")
        rel_error = (
            ((estimate - targets_array) + 1) / (targets_array + 1)
        ) ** 2
        if torch.isnan(rel_error).any():
            raise ValueError("Relative error contains NaNs")
        return rel_error.mean()

    def dropout_weights(weights, p):
        if p == 0:
            return weights
        # Replace p% of the weights with the mean value of the rest of them
        mask = torch.rand_like(weights) < p
        mean = weights[~mask].mean()
        masked_weights = weights.clone()
        masked_weights[mask] = mean
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=1e-1)
    from tqdm import trange

    start_loss = None

    iterator = trange(10_000)
    for i in iterator:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, dropout_rate)
        l = loss(torch.exp(weights_))
        if start_loss is None:
            start_loss = l.item()
        loss_rel_change = (l.item() - start_loss) / start_loss
        l.backward()
        iterator.set_postfix(
            {"loss": l.item(), "loss_rel_change": loss_rel_change}
        )
        optimizer.step()

    return torch.exp(weights).detach().numpy()


if __name__ == "__main__":
    ReweightedFRS_2022_23().generate()
    EnhancedFRS_2022_23().generate()
