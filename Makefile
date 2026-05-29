.PHONY: lock install test build

REQ_DIRS := requirements

lock:
	@set -e; \
	for d in $(REQ_DIRS); do \
		echo "Locking $$d ..."; \
		pip-compile $$d/requirements.in -o $$d/requirements.txt --generate-hashes --allow-unsafe; \
	done

install:
	pip install -r requirements/requirements.txt

test:
	python -m unittest discover -v

build:
	python -m build --wheel
