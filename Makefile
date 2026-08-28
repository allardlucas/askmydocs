.PHONY: install install-dev test lint format run-api run-ui eval clean

PYTHON := python
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check src/ tests/ eval/
	mypy src/

format:
	ruff check --fix src/ tests/ eval/
	ruff format src/ tests/ eval/

eval:
	$(PYTHON) -m eval.run_eval

run-api:
	$(PYTHON) -m uvicorn askmydocs.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	$(PYTHON) -m streamlit run src/askmydocs/ui/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
