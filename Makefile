.venv:
	uv venv -p 3.11

test:
	pytest tests -v

data/ukda:
	python data/download_private_prerequisites.py

data/models:
	python src/policyengine_uk_data_v2/imputations/income.py
	python src/policyengine_uk_data_v2/imputations/consumption.py
	python src/policyengine_uk_data_v2/imputations/wealth.py
	python src/policyengine_uk_data_v2/imputations/vat.py

