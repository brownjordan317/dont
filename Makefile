PYTHON := python3
SRC    := src

.PHONY: help train train_route_skill train_avoid_skill train_manager test test_single_drone eval eval_skills

help:
	@echo "Available targets:"
	@echo "  make train            - Auto-bootstrap missing HRL skills, then train manager"
	@echo "  make train_route_skill - Train the route-follow skill only"
	@echo "  make train_avoid_skill - Train the avoidance skill only"
	@echo "  make train_manager     - Train the manager only"
	@echo "  make test      - Run test and Inference"
	@echo "  make test_single_drone - Run a one-drone route-skill test"
	@echo "  make eval      - Evaluate manager model"
	@echo "  make eval_skills - Individually evaluate route/avoid skills with numbers and visuals"

train:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train

train_route_skill:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train_route_skill

train_avoid_skill:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train_avoid_skill

train_manager:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train_manager

test:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test

test_single_drone:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test --single-drone

eval:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode eval

eval_skills:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode eval_skills
