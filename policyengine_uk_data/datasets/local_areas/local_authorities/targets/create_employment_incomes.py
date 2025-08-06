import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore")

income = pd.read_excel(
    "nomis_earning_jobs_data.xlsx"
)  # Ensure only one ".xlsx" extension
# Export the above from Nomis- containing percentiles of earnings for each local authority (all jobs).
income = income.drop(index=range(0, 7)).reset_index(drop=True)
income.columns = income.iloc[0]
income = income.drop(index=0).reset_index(drop=True)
columns = list(income.columns)
for i, col in enumerate(columns):
    if pd.isna(col):
        columns[i] = "LA_code"
        break
income.columns = columns
columns_to_keep = [
    "local authority: district / unitary (as of April 2023)",
    "LA_code",
    "Number of jobs",
    "Median",
    "10 percentile",
    "20 percentile",
    "30 percentile",
    "40 percentile",
    "60 percentile",
    "70 percentile",
    "80 percentile",
    "90 percentile",
]
income = income[columns_to_keep]
income = income.rename(columns={"Median": "50 percentile"})
income = income.drop(index=0).reset_index(drop=True)
column_order = [
    "local authority: district / unitary (as of April 2023)",
    "LA_code",
    "Number of jobs",
    "10 percentile",
    "20 percentile",
    "30 percentile",
    "40 percentile",
    "50 percentile",
    "60 percentile",
    "70 percentile",
    "80 percentile",
    "90 percentile",
]
income = income[column_order]
income = income.replace("#", np.nan)

# reference values from here: https://www.gov.uk/government/statistics/percentile-points-from-1-to-99-for-total-income-before-and-after-tax#:~:text=Details,in%20the%20Background%20Quality%20Report.
reference_values = {
    10: 15300,
    20: 18000,
    30: 20800,
    40: 23700,
    50: 27200,
    60: 31600,
    70: 37500,
    80: 46100,
    90: 62000,
    91: 65300,
    92: 69200,
    93: 74000,
    94: 79800,
    95: 87400,
    96: 97200,
    97: 111000,
    98: 137000,
    100: 199000,
}

# List of columns to work on, now including 91-98 percentiles, in order
percentile_columns = [
    "10 percentile",
    "20 percentile",
    "30 percentile",
    "40 percentile",
    "50 percentile",
    "60 percentile",
    "70 percentile",
    "80 percentile",
    "90 percentile",
    "91 percentile",
    "92 percentile",
    "93 percentile",
    "94 percentile",
    "95 percentile",
    "96 percentile",
    "97 percentile",
    "98 percentile",
    "100 percentile",
]

# Ensure all new percentile columns exist in the DataFrame, set them to NaN initially if they don’t
for col in percentile_columns:
    if col not in income.columns:
        income[col] = np.nan

# Convert all percentile columns to numeric, coercing errors to NaN
income[percentile_columns] = income[percentile_columns].apply(
    pd.to_numeric, errors="coerce"
)


# Function to fill missing values based on reference ratios
def fill_missing_percentiles(row):
    # Extract known values and their corresponding percentiles
    known_values = {
        int(col.split()[0]): row[col]
        for col in percentile_columns
        if pd.notna(row[col])
    }

    # If no values are known, return the row unchanged
    if not known_values:
        return row

    # Sort known values by percentile to calculate intermediate values
    known_percentiles = sorted(known_values.keys())

    # Fill missing values based on ratios
    for col in percentile_columns:
        percentile = int(col.split()[0])

        # If this percentile is missing in the row
        if pd.isna(row[col]):
            # Find the closest lower and upper known percentiles
            lower = max(
                [p for p in known_percentiles if p < percentile], default=None
            )
            upper = min(
                [p for p in known_percentiles if p > percentile], default=None
            )

            # If both lower and upper bounds exist, interpolate
            if lower is not None and upper is not None:
                # Ratio between the target percentile and the lower bound
                lower_ratio = (
                    reference_values[percentile] / reference_values[lower]
                )
                row[col] = row[f"{lower} percentile"] * lower_ratio

            # If only the lower bound exists, extrapolate upwards
            elif lower is not None:
                lower_ratio = (
                    reference_values[percentile] / reference_values[lower]
                )
                row[col] = row[f"{lower} percentile"] * lower_ratio

            # If only the upper bound exists, extrapolate downwards
            elif upper is not None:
                upper_ratio = (
                    reference_values[percentile] / reference_values[upper]
                )
                row[col] = row[f"{upper} percentile"] * upper_ratio

    return row


# Apply the function to each row in the DataFrame
income = income.apply(fill_missing_percentiles, axis=1)

# Reorder columns with the percentiles at the end
non_percentile_columns = [
    col for col in income.columns if col not in percentile_columns
]
income = income[non_percentile_columns + percentile_columns]

# Display the updated DataFrame
# income.head()

# Set up the '0 percentile' column in the DataFrame
income["0 percentile"] = 0

# Ensure all percentile columns are numeric, converting any non-numeric values to NaN
percentile_columns = [
    "0 percentile",
    "10 percentile",
    "20 percentile",
    "30 percentile",
    "40 percentile",
    "50 percentile",
    "60 percentile",
    "70 percentile",
    "80 percentile",
    "90 percentile",
    "91 percentile",
    "92 percentile",
    "93 percentile",
    "94 percentile",
    "95 percentile",
    "96 percentile",
    "97 percentile",
    "98 percentile",
    "100 percentile",
]

for col in percentile_columns:
    income[col] = pd.to_numeric(income[col], errors="coerce")

# Define the updated income bands with infinity for the last band
income_bands = [
    (0, 12570),
    (12570, 15000),
    (15000, 20000),
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 70000),
    (70000, 100000),
    (100000, 150000),
    (150000, 200000),
    (200000, 300000),
    (300000, 500000),
    (500000, float("inf")),  # Use infinity for the last band
]


# Function to calculate population count for each income band
def calculate_band_population(row):
    # Define the percentiles and income values, including new percentiles 91 to 98
    percentiles = [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        100,
    ]
    income_values = [
        row["0 percentile"],
        row["10 percentile"],
        row["20 percentile"],
        row["30 percentile"],
        row["40 percentile"],
        row["50 percentile"],
        row["60 percentile"],
        row["70 percentile"],
        row["80 percentile"],
        row["90 percentile"],
        row["91 percentile"],
        row["92 percentile"],
        row["93 percentile"],
        row["94 percentile"],
        row["95 percentile"],
        row["96 percentile"],
        row["97 percentile"],
        row["98 percentile"],
        row["100 percentile"],
    ]

    # Filter out NaN values and corresponding percentiles
    filtered_percentiles = [
        p for p, v in zip(percentiles, income_values) if pd.notna(v)
    ]
    filtered_income_values = [v for v in income_values if pd.notna(v)]

    # If there are insufficient data points to create a spline, return zero population counts
    if len(filtered_percentiles) < 2:
        band_df = pd.DataFrame(
            income_bands, columns=["income_lower_bound", "income_upper_bound"]
        )
        band_df["population_count"] = [0] * len(income_bands)
        band_df["local authority: district / unitary (as of April 2023)"] = (
            row["local authority: district / unitary (as of April 2023)"]
        )
        band_df["LA_code"] = row["LA_code"]
        return band_df

    # Fit a linear spline to the available data
    spline = interp1d(
        filtered_percentiles,
        filtered_income_values,
        kind="linear",
        fill_value="extrapolate",
    )

    # Total jobs for this local authority (equivalent to total population)
    total_jobs = row["Number of jobs"]

    # Calculate the population count for each income band
    band_population_counts = []
    for lower, upper in income_bands:
        # Find the approximate percentile range for the income band
        lower_percentile = np.interp(
            lower, filtered_income_values, filtered_percentiles
        )
        upper_percentile = np.interp(
            upper, filtered_income_values, filtered_percentiles
        )

        # Ensure lower_percentile is less than upper_percentile
        if lower_percentile < upper_percentile:
            # Integrate to get proportion in this income band
            proportion_in_band, _ = quad(
                spline, lower_percentile, upper_percentile
            )
            proportion_in_band = proportion_in_band / spline(
                filtered_percentiles[-1]
            )  # Normalize by max spline value
            population_in_band = proportion_in_band * total_jobs
        else:
            population_in_band = 0  # No population if band is out of range

        band_population_counts.append(population_in_band)

    # Adjust to ensure the sum matches the total jobs, if possible
    total_estimated_population = sum(band_population_counts)
    if total_estimated_population > 0:
        scaling_factor = total_jobs / total_estimated_population
        band_population_counts = [
            count * scaling_factor for count in band_population_counts
        ]
    else:
        band_population_counts = [0] * len(
            income_bands
        )  # Set all to zero if no population is estimated

    # Create a result DataFrame for this row
    band_df = pd.DataFrame(
        income_bands, columns=["income_lower_bound", "income_upper_bound"]
    )
    band_df["population_count"] = band_population_counts
    band_df["local authority: district / unitary (as of April 2023)"] = row[
        "local authority: district / unitary (as of April 2023)"
    ]
    band_df["LA_code"] = row["LA_code"]

    return band_df


# Apply the function to each row and concatenate results
result_df = pd.concat(
    income.apply(calculate_band_population, axis=1).to_list(),
    ignore_index=True,
)

# Display the final DataFrame
result_df = result_df[
    [
        "local authority: district / unitary (as of April 2023)",
        "LA_code",
        "income_lower_bound",
        "income_upper_bound",
        "population_count",
    ]
]

result_df["total_earning"] = result_df["population_count"] * (
    (
        np.where(
            result_df["income_upper_bound"] == np.inf,
            1000000,
            result_df["income_upper_bound"],
        )
        + result_df["income_lower_bound"]
    )
    / 2
)

result_df = result_df.dropna(subset=["LA_code"])

total_income = pd.read_csv("total_income.csv")

missing_codes = total_income[~total_income["code"].isin(result_df["LA_code"])][
    "code"
].unique()

# Assuming result_df and total_income DataFrames are loaded
# Step 1: Identify missing codes
missing_codes = total_income[~total_income["code"].isin(result_df["LA_code"])][
    "code"
].unique()

# Step 2: Make a copy of result_df to work on
result_df_copy = result_df.copy()

# Step 3: Define income bounds based on the example structure
income_bounds = [
    (0, 12570),
    (12570, 15000),
    (15000, 20000),
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 70000),
    (70000, 100000),
    (100000, 150000),
    (150000, 200000),
    (200000, 300000),
    (300000, 500000),
    (500000, np.inf),
]

# Step 4: Create rows for each missing code
new_rows = []
for code in missing_codes:
    for lower, upper in income_bounds:
        new_rows.append(
            {
                "local authority: district / unitary (as of April 2023)": np.nan,  # Set to NaN (missing)
                "LA_code": code,
                "income_lower_bound": lower,
                "income_upper_bound": upper,
                "population_count": 0,
                "total_earning": 0,
            }
        )

# Step 5: Add the new rows to the DataFrame
missing_df = pd.DataFrame(new_rows)
result_df_copy = pd.concat([result_df_copy, missing_df], ignore_index=True)
result_df_copy = result_df_copy.dropna(subset=["LA_code"])
# result_df_copy.tail()

import numpy as np


def find_and_replace_zero_populations(
    result_df_copy, total_income
) -> pd.DataFrame:
    # Step 1: Find local authorities with all zero populations
    LA_with_zero_population = (
        result_df_copy.groupby("LA_code")
        .filter(lambda group: (group["population_count"] == 0).all())[
            "LA_code"
        ]
        .unique()
    )

    # Create a copy to avoid modifying the original DataFrame
    result_df = result_df_copy.copy()

    # Step 2: Process each zero-population local authority
    for zero_LA in LA_with_zero_population:
        try:
            # Get the total_income_count for the current local authority
            current_LA_data = total_income[total_income["code"] == zero_LA]

            if current_LA_data.empty:
                print(
                    f"Warning: No data found for local authority {zero_LA} in total_income"
                )
                continue

            current_total_income = current_LA_data[
                "total_income_count"
            ].values[0]

            # Find the nearest local authority by total_income_count
            # Exclude both the current local authority and other zero population local authorities
            other_LA = total_income[
                ~total_income["code"].isin(LA_with_zero_population)
            ]

            if other_LA.empty:
                print(
                    f"Warning: No valid local authorities found to copy from"
                )
                continue

            # Calculate absolute differences
            differences = np.abs(
                other_LA["total_income_count"] - current_total_income
            )

            # Get the index of the minimum difference
            min_diff_idx = differences.values.argmin()
            nearest_LA = other_LA.iloc[min_diff_idx]["code"]

            # Step 3: Copy population and earnings data from nearest local authority
            # For each income band of the zero local authority
            zero_const_rows = result_df[result_df["LA_code"] == zero_LA]

            if zero_const_rows.empty:
                print(
                    f"Warning: No rows found for local authority {zero_LA} in result_df"
                )
                continue

            for _, zero_row in zero_const_rows.iterrows():
                # Find the matching income band in the nearest local authority
                matching_row = result_df[
                    (result_df["LA_code"] == nearest_LA)
                    & (
                        result_df["income_lower_bound"]
                        == zero_row["income_lower_bound"]
                    )
                    & (
                        result_df["income_upper_bound"]
                        == zero_row["income_upper_bound"]
                    )
                ]

                if matching_row.empty:
                    print(
                        f"Warning: No matching income band found for local authority {nearest_LA}"
                    )
                    continue

                # Update the specific row and income band with the corresponding values
                mask = (
                    (result_df["LA_code"] == zero_LA)
                    & (
                        result_df["income_lower_bound"]
                        == zero_row["income_lower_bound"]
                    )
                    & (
                        result_df["income_upper_bound"]
                        == zero_row["income_upper_bound"]
                    )
                )

                result_df.loc[mask, "population_count"] = matching_row[
                    "population_count"
                ].values[0]
                result_df.loc[mask, "total_earning"] = matching_row[
                    "total_earning"
                ].values[0]

        except Exception as e:
            print(f"Error processing local authority {zero_LA}: {str(e)}")
            continue

    return result_df


updated_df = find_and_replace_zero_populations(result_df_copy, total_income)

updated_df = updated_df.rename(
    columns={
        "LA_code": "code",
        "local authority: district / unitary (as of April 2023)": "name",
        "income_lower_bound": "employment_income_lower_bound",
        "income_upper_bound": "employment_income_upper_bound",
        "population_count": "employment_income_count",
        "total_earning": "employment_income_amount",
    }
)[
    [
        "code",
        "name",
        "employment_income_lower_bound",
        "employment_income_upper_bound",
        "employment_income_count",
        "employment_income_amount",
    ]
]
updated_df = updated_df.sort_values("code")
updated_df.to_csv("employment_income.csv", index=False)
