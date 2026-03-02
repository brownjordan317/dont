PYTHON := python3
SRC    := src

.PHONY: help train test physics

help:
	@echo "Available targets:"
	@echo "  make train     - Run training"
	@echo "  make test      - Run test"

train:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train

test:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test
