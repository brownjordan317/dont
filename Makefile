PYTHON := python3
SRC    := src

.PHONY: help train test physics

help:
	@echo "Available targets:"
	@echo "  make train     - Run training"
	@echo "  make test      - Run test"
	@echo "  make test_light      - Run test without saving visuals"

train:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train

test:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test

test_light:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test_light
