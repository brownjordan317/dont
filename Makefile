PYTHON := python3
SRC    := src

.PHONY: help train test physics

help:
	@echo "Available targets:"
	@echo "  make train     - Run training"
	@echo "  make test      - Run test"
	@echo "  make physics   - Run physics tests"

train:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train

test:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test

physics:
	$(PYTHON) $(SRC)/test_physics.py
