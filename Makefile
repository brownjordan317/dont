PYTHON := python3
SRC    := src

.PHONY: help train test eval

help:
	@echo "Available targets:"
	@echo "  make train     - Run training"
	@echo "  make test      - Run test and Inference"
	@echo "  make eval      - Evaluate models"

train:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train

test:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test

eval:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode eval
