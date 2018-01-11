lint: clean
	bash tests/ensure_flake8.sh
	flake8 bonfire --format=pylint || true
test: lint
	bash tests/ensure_pytest.sh
	py.test -vvv --cov bonfire --cov-report term-missing --cov-report xml:cobertura.xml --junitxml=testresult.xml tests

clean:
	- find . -iname "*__pycache__" | xargs rm -rf
	- find . -iname "*.pyc" | xargs rm -rf
	- rm cobertura.xml -f
	- rm testresult.xml -f
	- rm .coverage -f

venv:
	- virtualenv --python=$(shell which python3) --prompt '<venv:bonfire>' venv

deps: venv
	- venv/bin/pip install -U pip setuptools
	- venv/bin/pip install -r requirements.txt
