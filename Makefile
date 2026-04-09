PYTHON ?= python3
CONFIG ?= config.yaml
MAX_STEPS ?= 2000
STOP_WHEN_COVERED_PERCENT ?=
SAVE_VIDEO ?=
OUTPUT ?= exports
IMAGE ?=
SOURCE_IMAGE_PPM ?=
TL_LAT ?=
TL_LON ?=
TR_LAT ?=
TR_LON ?=
BR_LAT ?=
BR_LON ?=
BL_LAT ?=
BL_LON ?=
SEARCH_DECAY_PERCENT_PER_100MS ?=
STEP_SECONDS ?=
IN_DIR ?=
OUT_DIR ?=

.PHONY: run compile export export-paths

EXPORT_ARGS = --config $(CONFIG) --max-steps $(MAX_STEPS)
ifneq ($(strip $(STOP_WHEN_COVERED_PERCENT)),)
EXPORT_ARGS += --stop-when-covered-percent $(STOP_WHEN_COVERED_PERCENT)
endif
ifneq ($(strip $(SAVE_VIDEO)),)
ifeq ($(strip $(SAVE_VIDEO)),true)
EXPORT_ARGS += --save-video
else ifeq ($(strip $(SAVE_VIDEO)),false)
EXPORT_ARGS += --no-video
else
$(error SAVE_VIDEO must be true or false when set)
endif
endif
ifneq ($(strip $(OUTPUT)),)
EXPORT_ARGS += --output $(OUTPUT)
endif
ifneq ($(strip $(IMAGE)),)
EXPORT_ARGS += --image $(IMAGE)
endif
ifneq ($(strip $(SOURCE_IMAGE_PPM)),)
EXPORT_ARGS += --source-image-ppm $(SOURCE_IMAGE_PPM)
endif
ifneq ($(strip $(SEARCH_DECAY_PERCENT_PER_100MS)),)
EXPORT_ARGS += --search-decay-percent-per-100ms $(SEARCH_DECAY_PERCENT_PER_100MS)
endif
ifneq ($(strip $(STEP_SECONDS)),)
EXPORT_ARGS += --step-seconds $(STEP_SECONDS)
endif
ifneq ($(strip $(TL_LAT)),)
EXPORT_ARGS += --tl-lat $(TL_LAT) --tl-lon $(TL_LON)
EXPORT_ARGS += --tr-lat $(TR_LAT) --tr-lon $(TR_LON)
EXPORT_ARGS += --br-lat $(BR_LAT) --br-lon $(BR_LON)
EXPORT_ARGS += --bl-lat $(BL_LAT) --bl-lon $(BL_LON)
endif

run:
	$(PYTHON) -m src.main

export: export-paths

export-paths:
	$(PYTHON) -m src.export_paths_geojson $(EXPORT_ARGS)

conv_mpms:
	@if [ -z "$(IN_DIR)" ] || [ -z "$(OUT_DIR)" ]; then \
		echo "ERROR: Both IN_DIR and OUT_DIR must be provided."; \
		echo "Usage: make conv_mpms IN_DIR=./raw_data OUT_DIR=./processed_data"; \
		exit 1; \
	fi
	$(PYTHON) -m src.mpms_conv --in-dir $(IN_DIR) --out-dir $(OUT_DIR)

compile:
	$(PYTHON) -m compileall src/
