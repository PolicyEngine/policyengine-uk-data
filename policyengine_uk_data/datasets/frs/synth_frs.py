import numpy as np


def create_synth_frs():
    from policyengine_uk import Microsimulation
    from policyengine_uk_data.datasets import EnhancedFRS_2022_23
    from policyengine_uk_data.storage import STORAGE_FOLDER

    simulation = Microsimulation(
        dataset=EnhancedFRS_2022_23,
    )

    data = {}
    for variable in simulation.tax_benefit_system.variables:
        data[variable] = {}
        for time_period in simulation.get_holder(variable).get_known_periods():
            values = simulation.get_holder(variable).get_array(time_period)
            values = np.array(values)
            if "_id" in variable:
                pass
            else:
                if (
                    simulation.tax_benefit_system.variables[
                        variable
                    ].value_type
                    in (float,)
                    and "_id" not in variable
                ):
                    # Add random noise
                    noise = (
                        np.random.random(size=values.shape)
                        * 0.1
                        * np.abs(values)
                    )
                    values = values + noise
                    # Shuffle values
                np.random.shuffle(values)
            if values is not None:
                data[variable][time_period] = values
        if len(data[variable]) == 0:
            del data[variable]

    import h5py

    with h5py.File(STORAGE_FOLDER / "synthetic_frs_2022_23.h5", "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)
