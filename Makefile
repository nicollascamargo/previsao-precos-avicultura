.PHONY: install demo dados backtest train figuras test lint api docker clean

install:
	pip install -e ".[dev,stats]"

## End-to-end: synthetic data -> backtest -> trained artifacts
demo: dados backtest train

dados:
	python -m precos_avicultura.cli dados --sintetico

backtest:
	python -m precos_avicultura.cli backtest

train:
	python -m precos_avicultura.cli treinar

figuras:
	python scripts/gerar_figuras.py

test:
	pytest --cov=precos_avicultura --cov-report=term-missing

lint:
	ruff check src tests

api:
	uvicorn precos_avicultura.api.main:app --reload --port 8000

docker:
	docker build -t precos-avicultura .
	docker run --rm -p 8000:8000 precos-avicultura

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .coverage
	find . -name "*.pyc" -delete
