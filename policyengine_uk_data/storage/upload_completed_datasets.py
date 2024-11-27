from policyengine_uk_data.datasets import EnhancedFRS_2022_23, FRS_2022_23

datasets = [EnhancedFRS_2022_23, FRS_2022_23]

for dataset in datasets:
    ds = dataset()
    print(f"Uploading {ds.name} with url {ds.url}")
    ds.upload()