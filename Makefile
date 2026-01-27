all: data test

format:
	black . -l 79

test:
	pytest .

install:
	uv pip install -e ".[dev]" --config-settings editable_mode=compat

# Dagster commands
ui:
	dagster dev -m policyengine_uk_data.definitions

dev: ui

data:
	dagster asset materialize --select "*" -m policyengine_uk_data.definitions

data-test:
	TESTING=1 dagster asset materialize --select "*" -m policyengine_uk_data.definitions

# Materialise specific asset groups
raw:
	dagster asset materialize --select "group:raw_data" -m policyengine_uk_data.definitions

models:
	dagster asset materialize --select "group:models" -m policyengine_uk_data.definitions

imputations:
	dagster asset materialize --select "group:imputations" -m policyengine_uk_data.definitions

calibration:
	dagster asset materialize --select "group:calibration" -m policyengine_uk_data.definitions

output:
	dagster asset materialize --select "enhanced_frs" -m policyengine_uk_data.definitions

targets:
	dagster asset materialize --select "targets_db" -m policyengine_uk_data.definitions

build:
	python -m build

publish:
	twine upload dist/*

changelog:
	build-changelog changelog.yaml --output changelog.yaml --update-last-date --start-from 1.0.0 --append-file changelog_entry.yaml
	build-changelog changelog.yaml --org PolicyEngine --repo policyengine-uk-data --output CHANGELOG.md --template .github/changelog_template.md
	bump-version changelog.yaml pyproject.toml
	rm changelog_entry.yaml || true
	touch changelog_entry.yaml

# Dashboard commands (local - faster)
dashboard:
	@echo "Starting dashboard locally..."
	@cd dashboard/api && DATABASE_PATH=../../policyengine_uk_data/targets/targets.db CORS_ORIGINS=http://localhost:3000 uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 & \
	cd dashboard/frontend && bun dev > /tmp/frontend.log 2>&1 & \
	sleep 3 && \
	echo "✓ Frontend: http://localhost:3000" && \
	echo "✓ API: http://localhost:8000" && \
	echo "✓ API Docs: http://localhost:8000/docs"

dashboard-stop:
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@echo "Dashboard stopped"

# Dashboard commands (Docker - slower but isolated)
dashboard-docker:
	cd dashboard && docker-compose up --build

dashboard-docker-down:
	cd dashboard && docker-compose down

# Help
help:
	@echo "Available targets:"
	@echo "  make ui          - Start Dagster web UI"
	@echo "  make data        - Materialise all assets"
	@echo "  make data-test   - Materialise all assets (testing mode, reduced epochs)"
	@echo "  make raw         - Materialise raw data assets only"
	@echo "  make models      - Materialise imputation models only"
	@echo "  make imputations - Materialise imputation assets only"
	@echo "  make calibration - Materialise calibration assets only"
	@echo "  make output      - Materialise final enhanced_frs only"
	@echo "  make targets     - Materialise targets database only"
	@echo "  make dashboard         - Start dashboard locally (fast)"
	@echo "  make dashboard-stop    - Stop dashboard"
	@echo "  make dashboard-docker  - Start dashboard with Docker (slower)"
	@echo "  make test        - Run pytest"
	@echo "  make format      - Format with black"
	@echo "  make install     - Install package with dev deps"
