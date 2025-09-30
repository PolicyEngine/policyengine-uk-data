# API Reference

This page documents the main classes and functions in PolicyEngine UK Data.

## Datasets

### Main Datasets

All datasets inherit from `policyengine_uk.data.UKSingleYearDataset` and can be used with `policyengine_uk.Microsimulation`.

#### `FRS_2022_23`

Raw Family Resources Survey data for 2022-23.

**Use case:** Baseline comparison, replicating official FRS analysis

**Features:**
- Demographics from FRS
- Reported benefits (not simulated)
- No wealth or consumption data
- Known income underreporting

```python
from policyengine_uk_data import FRS_2022_23
from policyengine_uk import Microsimulation

simulation = Microsimulation(dataset=FRS_2022_23)
```

#### `ExtendedFRS_2022_23`

FRS with imputed wealth and consumption variables.

**Use case:** Analysis requiring wealth or consumption but not requiring maximum income accuracy

**Additions over FRS:**
- Wealth variables (from WAS)
- Consumption variables (from LCFS)
- VAT exposure rates (from ETB)
- Simulated benefits (not reported)

```python
from policyengine_uk_data import ExtendedFRS_2022_23

simulation = Microsimulation(dataset=ExtendedFRS_2022_23)
```

####

 `EnhancedFRS_2022_23` (Recommended)

Extended FRS with SPI-based income enhancement to correct high-income underreporting.

**Use case:** Most policy analysis (recommended default)

**Additions over Extended FRS:**
- High-income correction using SPI data
- More accurate income distribution
- Maintains all wealth/consumption imputations

```python
from policyengine_uk_data import EnhancedFRS_2022_23

simulation = Microsimulation(dataset=EnhancedFRS_2022_23)
```

#### `ReweightedFRS_2022_23`

Enhanced FRS with calibrated weights to match official statistics.

**Use case:** Maximum accuracy, official statistic replication

**Additions over Enhanced FRS:**
- Calibrated to 2000+ official statistics
- Matches HMRC, DWP, OBR data
- Higher computational cost

```python
from policyengine_uk_data import ReweightedFRS_2022_23

simulation = Microsimulation(dataset=ReweightedFRS_2022_23)
```

### Local Area Datasets

#### `Constituency_2024_25`

Parliamentary constituency-level dataset.

```python
from policyengine_uk_data.datasets.local_areas import Constituency_2024_25

simulation = Microsimulation(dataset=Constituency_2024_25)
constituency = simulation.calculate("constituency", period=2025)
```

#### `LocalAuthority_2024_25`

Local authority-level dataset.

```python
from policyengine_uk_data.datasets.local_areas import LocalAuthority_2024_25

simulation = Microsimulation(dataset=LocalAuthority_2024_25)
local_authority = simulation.calculate("local_authority", period=2025)
```

## Utility Functions

### Dataset Utilities

#### `sum_to_entity(df, entity, variable, target_entity)`

Aggregate a variable from one entity level to another.

**Parameters:**
- `df` (DataFrame): Source data
- `entity` (str): Source entity level
- `variable` (str): Variable to aggregate
- `target_entity` (str): Target entity level

**Returns:** Aggregated series

```python
from policyengine_uk_data.utils.datasets import sum_to_entity

# Sum person-level income to household level
household_income = sum_to_entity(
    df=person_df,
    entity="person",
    variable="employment_income",
    target_entity="household"
)
```

#### `categorical(series, categories)`

Convert a series to categorical codes.

**Parameters:**
- `series` (Series): Input series
- `categories` (dict): Mapping of values to category codes

**Returns:** Series with categorical codes

### Loss/Validation Functions

#### `get_loss_results(dataset, time_period, reform=None)`

Calculate validation metrics comparing dataset to official statistics.

**Parameters:**
- `dataset` (UKSingleYearDataset): Dataset to validate
- `time_period` (int): Year to validate
- `reform` (Reform, optional): Policy reform to apply

**Returns:** DataFrame with validation metrics including:
- `name`: Statistic name
- `estimate`: Dataset estimate
- `target`: Official statistic
- `error`: Absolute error
- `rel_error`: Relative error
- `abs_rel_error`: Absolute relative error

```python
from policyengine_uk_data.utils import get_loss_results
from policyengine_uk_data import EnhancedFRS_2022_23

results = get_loss_results(EnhancedFRS_2022_23, 2025)
print(f"Mean absolute relative error: {results.abs_rel_error.mean():.2%}")
```

### Download Functions

#### `download_prerequisites()`

Download required data files from Hugging Face.

**Requires:** `HUGGING_FACE_TOKEN` environment variable

```python
from policyengine_uk_data import download_prerequisites

download_prerequisites()
```

#### `check_prerequisites()`

Check if required data files are present.

**Returns:** Boolean indicating if all prerequisites are available

```python
from policyengine_uk_data import check_prerequisites

if not check_prerequisites():
    print("Missing prerequisites. Run download_prerequisites()")
```

## Constants

### `STORAGE_FOLDER`

Path to local data storage directory.

```python
from policyengine_uk_data.utils.datasets import STORAGE_FOLDER

print(f"Data stored in: {STORAGE_FOLDER}")
```

## Building Custom Datasets

### `create_frs(raw_frs_folder, year)`

Process raw FRS data into PolicyEngine format.

**Parameters:**
- `raw_frs_folder` (str): Path to raw FRS `.tab` files
- `year` (int): Survey year

**Returns:** `UKSingleYearDataset`

```python
from policyengine_uk_data.datasets.frs import create_frs

dataset = create_frs(
    raw_frs_folder="/path/to/frs/data",
    year=2022
)
```

### Imputation Modules

Located in `policyengine_uk_data.datasets.imputations`:

- `wealth` - Wealth variable imputations from WAS
- `income` - Income enhancements from SPI
- `consumption` - Consumption imputations from LCFS
- `vat` - VAT exposure rates from ETB
- `capital_gains` - Capital gains imputations

Each module provides functions to add imputed variables to datasets.

## See Also

- [Getting Started](getting_started.md) - Installation and basic usage
- [Examples](examples.md) - Detailed usage examples
- [Methodology](methodology.ipynb) - How datasets are created