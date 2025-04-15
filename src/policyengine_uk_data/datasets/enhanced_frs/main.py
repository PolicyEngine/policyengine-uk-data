from policyengine_uk_data.datasets.frs import FRS
from pathlib import Path
from policyengine_uk_data.datasets.enhanced_frs.imputations import add_imputations, add_income_imputations


class EFRS(FRS):
    def generate(self, year: int, frs_checkpoint: str | Path = None):
        self.year = year
        if frs_checkpoint is None:
            super().generate(year=year)
        else:
            self.load_from_h5(frs_checkpoint, year=year)

        add_imputations(self)
        add_income_imputations(self)
        self.save_to_h5("efrs_2022.h5")

    


    
