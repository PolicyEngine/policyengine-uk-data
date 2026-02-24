all: data test

format:
	black . -l 79

test:
	pytest .

install:
	uv pip install -e ".[dev]" --config-settings editable_mode=compat

download:
	python policyengine_uk_data/storage/download_private_prerequisites.py

upload:
	python policyengine_uk_data/storage/upload_completed_datasets.py

documentation:
	pip install --pre "jupyter-book>=2"
	jb clean docs && jb build docs
	python docs/add_plotly_to_book.py docs

data:
	python policyengine_uk_data/datasets/create_datasets.py

build:
	python -m build

publish:
	twine upload dist/*

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")