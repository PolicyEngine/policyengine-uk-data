all: data test

format:
	black . -l 79

test:
	pytest

install:
	pip install -e ".[dev]"

install-lite:
	pip install -e .

download:
	python policyengine_uk_data/storage/download_private_prerequisites.py

upload:
	python policyengine_uk_data/storage/upload_completed_datasets.py

docker:
	docker buildx build --platform linux/amd64 . -t policyengine-uk-data:latest

documentation:
	streamlit run docs/Home.py

data:
	python policyengine_uk_data/datasets/frs/dwp_frs.py
	python policyengine_uk_data/datasets/frs/frs.py

build:
	python -m build

publish:
	twine upload dist/*
