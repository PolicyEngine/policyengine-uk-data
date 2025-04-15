.venv:
	uv venv -p 3.11

test:
	pytest tests -v

data/ukda:
	python data/download_private_prerequisites.py

imputations:
	python src/policyengine_uk_data/datasets/enhanced_frs/imputations/income.py
	python src/policyengine_uk_data/datasets/enhanced_frs/imputations/consumption.py
	python src/policyengine_uk_data/datasets/enhanced_frs/imputations/wealth.py
	python src/policyengine_uk_data/datasets/enhanced_frs/imputations/vat.py
