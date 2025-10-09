all: data test

format:
	uv run black . -l 79

test:
	uv run pytest .

install:
	uv pip install -e ".[dev]" --config-settings editable_mode=compat

download:
	uv run python policyengine_uk_data/storage/download_private_prerequisites.py

upload:
	uv run python policyengine_uk_data/storage/upload_completed_datasets.py

documentation:
	uv pip install --pre "jupyter-book>=2"
	jb clean docs && jb build docs
	uv run python docs/add_plotly_to_book.py docs

data:
	uv run python policyengine_uk_data/datasets/create_datasets.py

build:
	uv run python -m build

publish:
	twine upload dist/*

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 1.0.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-us-data --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml
