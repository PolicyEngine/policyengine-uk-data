import pandas as pd
import numpy as np
from pathlib import Path

# 1. Read the boundary change data into a DataFrame
df = pd.read_csv(Path(__file__).parent / "boundary_changes.csv")

# 2. Get unique constituency codes for 2010 and 2024, sorted alphabetically
old_codes = sorted(df["code_2010"].unique())
new_codes = sorted(df["code_2024"].unique())

# 3. Create index maps for old and new codes
old_index = {code: idx for idx, code in enumerate(old_codes)}
new_index = {code: idx for idx, code in enumerate(new_codes)}

# 4. Calculate proportion of old constituency's population in each new constituency
#    First, compute total population of each old constituency using groupby and transform
total_old_pop = df.groupby("code_2010")["old_population_present"].transform("sum")
#    Then compute the proportion for each row
df["proportion"] = df["old_population_present"] / total_old_pop

# 5. Initialize the transformation matrix M with zeros (dimensions 650 x 650)
mapping_matrix = np.zeros((len(old_codes), len(new_codes)), dtype=float)

# Prepare arrays of indices for fancy indexing assignment
old_idx_array = df["code_2010"].map(old_index).to_numpy()
new_idx_array = df["code_2024"].map(new_index).to_numpy()
proportions = df["proportion"].to_numpy()

# Use fancy indexing to assign proportions to the matrix in one operation
mapping_matrix[old_idx_array, new_idx_array] = proportions

# (Optional) Verify that each row sums to ~1 (allowing a tiny floating-point tolerance)
row_sums = mapping_matrix.sum(axis=1)
if not np.allclose(row_sums, 1.0):
    print("Warning: Not all rows sum to 1. Check data for consistency.")
