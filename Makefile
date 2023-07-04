clean-pyc:
	find . -name "*.pyc" | xargs rm -f

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count astronomical_matching; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

black:
	@if command -v black > /dev/null; then \
		echo "Running black"; \
		black --line-length 80 astronomical_matching; \
	else \
		echo "black not found, please install it!"; \
		exit 1; \
	fi;
	@echo "black passed"

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle astronomical_matching

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort astronomical_matching; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

style:
	@$(MAKE) isort
	@$(MAKE) black
	@$(MAKE) flake
	mypy ./astronomical_matching
	@$(MAKE) pydocstyle