import pandas as pd
import numpy as np

mapping_raw = pd.read_csv("mapping_2010_to_2024/mapping_raw.csv")
mapping_raw = mapping_raw.sort_values(["PCON10CD", "PCON24CD"])
mapping_raw = mapping_raw.reset_index(drop=True)

# Create sets of unique values for both columns
unique_pcon10 = mapping_raw["PCON10CD"].unique()
unique_pcon24 = mapping_raw["PCON24CD"].unique()

# Create an empty matrix filled with zeros
mapping_matrix = pd.DataFrame(0, index=unique_pcon10, columns=unique_pcon24)

# Fill the matrix using a for loop
for _, row in mapping_raw.iterrows():
    mapping_matrix.loc[row["PCON10CD"], row["PCON24CD"]] = 1

# Create sets of unique values for both columns
unique_pcon10 = mapping_raw["PCON10CD"].unique()
unique_pcon24 = mapping_raw["PCON24CD"].unique()

# Create empty matrix filled with zeros
mapping_matrix = pd.DataFrame(0, index=unique_pcon10, columns=unique_pcon24)

# Let's check the first constituency to see what's happening
example_pcon24 = unique_pcon24[0]
print(f"Example 2024 constituency: {example_pcon24}")

# Check if we can find it in mapping_raw
matching_rows = mapping_raw[mapping_raw["PCON24CD"] == example_pcon24]
print("\nMatching rows found:", len(matching_rows))
print(matching_rows)

# Now fill the matrix with proper checks
for pcon24 in unique_pcon24:
    # Get matching 2010 constituencies
    matching_2010 = mapping_raw[mapping_raw["PCON24CD"] == pcon24]["PCON10CD"]

    if len(matching_2010) > 0:  # Check if we found any matches
        weight = 1 / len(matching_2010)
        mapping_matrix.loc[matching_2010, pcon24] = weight
    else:
        print(f"No matches found for {pcon24}")

# Verify results
print("\nFirst few rows of result:")
print(mapping_matrix.head())
print("\nColumn sums (should be 1):")
print(mapping_matrix.sum().head())

# Show non-zero entries for first column
first_col = mapping_matrix[mapping_matrix.columns[0]]
print("\nNon-zero entries in first column:")
print(first_col[first_col > 0])

mapping_matrix.to_csv("mapping_2010_to_2024/mapping_matrix.csv")
