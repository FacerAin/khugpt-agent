clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-test-dep set-git set-dev
test: set-dev set-test-dep set-test
run: set-dev run-uvicorn
coverage: set-coverage


##### basic #####
set-git:
	git config --local commit.template .gitmessage

set-style-dep:
	pip3 install isort==5.12.0 black==23.3.0 flake8==4.0.1

set-test-dep:
	pip install -r test-requirements.txt

set-precommit:
	pip3 install pre-commit==2.17.0
	pre-commit install

set-dev:
	pip3 install -r requirements.txt

set-test:
	python3 -m pytest tests/ -m "not live"

set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache

run-uvicorn:
	uvicorn app.main:app

set-coverage:
	python3 -m pytest --cov-report term-missing --cov=app tests/ -m "not live"
