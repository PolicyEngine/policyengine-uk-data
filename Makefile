.venv:
	uv venv -p 3.11

test:
	pytest tests -v