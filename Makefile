clean-pyc:
	find . -name "*.pyc" | xargs rm -f

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne_icalabel examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

black:
	@if command -v black > /dev/null; then \
		echo "Running black"; \
		black mne_icalabel examples; \
	else \
		echo "black not found, please install it!"; \
		exit 1; \
	fi;
	@echo "black passed"

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne_icalabel

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort mne_icalabel examples doc; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

style:
	isort astronomical_matching
	black astronomical_matching
	flake8 astronomical_matching
	mypy ./astronomical_matching
	@$(MAKE) pydocstyle