all: data test

format:
	black . -l 79

test:
	pytest

install:
	pip install policyengine-uk==2.1.1
	pip install -e ".[dev]"

download:
	python policyengine_uk_data/storage/download_private_prerequisites.py

upload:
	python policyengine_uk_data/storage/upload_completed_datasets.py

docker:
	docker buildx build --platform linux/amd64 . -t policyengine-uk-data:latest

documentation:
	jb clean docs && jb build docs

data:
	python policyengine_uk_data/datasets/frs/dwp_frs.py
	python policyengine_uk_data/datasets/frs/frs.py
	python policyengine_uk_data/datasets/frs/extended_frs.py
	python policyengine_uk_data/datasets/frs/enhanced_frs.py

build:
	python -m build

publish:
	twine upload dist/*

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 1.0.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-us-data --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml