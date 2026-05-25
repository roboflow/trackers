# MOT benchmark workflow — Makefile orchestrates `trackers tune`, `trackers eval`,
# small helper scripts under scripts/, and `codabench_submit.py`.
#
#   cd benchmark
#   make help
#   make data-check
#   make benchmark-default TRACKER=sort CODABENCH_TOKEN=...
#   make benchmark-tuned   TRACKER=sort CODABENCH_TOKEN=...
#   make benchmark BENCHMARK_CONFIG=all TRACKER=sort CODABENCH_TOKEN=...
#
# See README.md for data setup.

SHELL     := /bin/bash
ROOT      := $(CURDIR)
REPO_ROOT := $(abspath $(ROOT)/..)
PYTHON    ?= python

DATA_ROOT  ?= $(ROOT)/data
PREP_DIR   ?= $(ROOT)/benchmark_prep
OUTPUT_DIR ?= $(ROOT)/benchmark_outputs

TRACKER   ?= sort
DATASET   ?= mot17
CONFIG    ?= default
BENCHMARK_CONFIG ?= default
N_TRIALS  ?= 10
OBJECTIVE ?= HOTA
THRESHOLD ?= 0.5
SEED      ?=
FIXED_PARAMS ?=

CODABENCH_URL           ?= https://www.codabench.org
CODABENCH_TOKEN         ?=
CODABENCH_WAIT_TIMEOUT  ?= 3600
CODABENCH_POLL_INTERVAL ?= 10

# BoT-SORT requires CMC on by default for the published numbers.
ifeq ($(TRACKER),botsort)
  ifeq ($(strip $(FIXED_PARAMS)),)
    FIXED_PARAMS := {"enable_cmc": true}
  endif
endif

CODABENCH_DATASETS := mot17 sportsmot dancetrack
LOCAL_DATASETS     := soccernet
ALL_DATASETS       := $(CODABENCH_DATASETS) $(LOCAL_DATASETS)
DATASETS           ?= $(ALL_DATASETS)

# Per-dataset tune splits. Score split is "test" for everyone (Codabench test for
# codabench datasets, public test GT for soccernet). Path lookups for image dirs
# delegate to scripts/datasets.py to stay DRY.
mot17_TUNE_SPLIT      := val
sportsmot_TUNE_SPLIT  := val
dancetrack_TUNE_SPLIT := train
soccernet_TUNE_SPLIT  := train
SCORE_SPLIT := test

# Codabench (competition_id, phase_id) per dataset.
mot17_CB      := 10049 16382
sportsmot_CB  := 13077 21402
dancetrack_CB := 14885 24635

LAYOUT := $(PYTHON) scripts/datasets.py --data-root "$(DATA_ROOT)"

.PHONY: help setup data-check prep prep-all tune track-default track-tuned _track-and-score \
        benchmark benchmark-default benchmark-tuned upload collect clean

help:
	@echo "MOT benchmark workflow — run from \`cd benchmark\`"
	@echo ""
	@echo "Targets:"
	@echo "  setup         Install \`trackers[tune]\` from $(REPO_ROOT)"
	@echo "  data-check    Print present/missing assets under $(DATA_ROOT)"
	@echo "  prep          Prep one dataset (DATASET=...) into $(PREP_DIR)"
	@echo "  prep-all      Prep every dataset"
	@echo "  tune          Tune one (TRACKER=, DATASET=)"
	@echo "  track-default     Track on default params, then score (TRACKER=, DATASET=)"
	@echo "  track-tuned       Track on best_params.json, then score (TRACKER=, DATASET=)"
	@echo "  upload            Upload an existing submission.zip (TRACKER=, DATASET=, CONFIG=default|tuned)"
	@echo "  benchmark         Full pipeline (TRACKER=, BENCHMARK_CONFIG=default|tuned|all, DATASETS=...)"
	@echo "  benchmark-default Same as \`make benchmark BENCHMARK_CONFIG=default\`"
	@echo "  benchmark-tuned   Same as \`make benchmark BENCHMARK_CONFIG=tuned\`"
	@echo "  collect           Aggregate eval/codabench JSONs into tables.md (TRACKER=)"
	@echo "  clean         Remove $(PREP_DIR) and $(OUTPUT_DIR)"
	@echo ""
	@echo "Codabench upload requires CODABENCH_TOKEN. See README for data setup."

setup:
	$(PYTHON) -m pip install -e "$(REPO_ROOT)[tune]"

data-check:
	@$(PYTHON) scripts/data_check.py --data-root "$(DATA_ROOT)"

prep:
	$(PYTHON) scripts/prep_data.py --dataset $(DATASET) --split all \
	  --data-root "$(DATA_ROOT)" --prep-dir "$(PREP_DIR)"

prep-all:
	$(PYTHON) scripts/prep_data.py --dataset all --split all \
	  --data-root "$(DATA_ROOT)" --prep-dir "$(PREP_DIR)"

# Tune via the `trackers tune` CLI. Pass --images-dir only for BoT-SORT CMC (ByteTrack/SORT/OC-SORT are det-only).
tune:
	@if [ ! -d "$(PREP_DIR)/$(DATASET)/$($(DATASET)_TUNE_SPLIT)" ]; then \
	  echo "Run: make prep DATASET=$(DATASET)"; exit 1; \
	fi
	@mkdir -p "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)"
	$(eval TUNE_SPLIT := $($(DATASET)_TUNE_SPLIT))
	$(eval TUNE_IMAGES := $(shell $(LAYOUT) --dataset $(DATASET) --split $(TUNE_SPLIT) --field images_dir))
	trackers tune \
	  --tracker $(TRACKER) \
	  --gt-dir "$(PREP_DIR)/$(DATASET)/$(TUNE_SPLIT)/gt" \
	  --detections-dir "$(PREP_DIR)/$(DATASET)/$(TUNE_SPLIT)/dets" \
	  --metrics CLEAR HOTA Identity --objective $(OBJECTIVE) --threshold $(THRESHOLD) \
	  --n-trials $(N_TRIALS) \
	  $(if $(SEED),--seed $(SEED),) \
	  $(if $(FIXED_PARAMS),--fixed-params '$(FIXED_PARAMS)',) \
	  $(if $(and $(filter botsort,$(TRACKER)),$(TUNE_IMAGES)),--images-dir "$(TUNE_IMAGES)",) \
	  --output "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/best_params.json"

track-default:
	@$(MAKE) _track-and-score TRACKER=$(TRACKER) DATASET=$(DATASET) CONFIG=default

track-tuned:
	@test -f "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/best_params.json" || \
	  { echo "missing best_params.json — run \`make tune TRACKER=$(TRACKER) DATASET=$(DATASET)\`"; exit 1; }
	@$(MAKE) _track-and-score TRACKER=$(TRACKER) DATASET=$(DATASET) CONFIG=tuned

# 1) track on the score split (test) → predictions
# 2) score: `trackers eval` for soccernet, `mot_format.py` + `codabench_submit.py` for the rest.
_track-and-score:
	@if [ ! -d "$(PREP_DIR)/$(DATASET)/$(SCORE_SPLIT)" ]; then \
	  echo "Run: make prep DATASET=$(DATASET)"; exit 1; \
	fi
	$(PYTHON) scripts/track_split.py \
	  --tracker $(TRACKER) --dataset $(DATASET) --split $(SCORE_SPLIT) \
	  --data-root "$(DATA_ROOT)" --prep-dir "$(PREP_DIR)" \
	  --output-dir "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)" \
	  $(if $(filter tuned,$(CONFIG)),--params "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/best_params.json",)
	@if [ "$(DATASET)" = "soccernet" ]; then \
	  trackers eval \
	    --gt-dir "$(PREP_DIR)/soccernet/test/gt" \
	    --tracker-dir "$(OUTPUT_DIR)/$(TRACKER)/soccernet/$(CONFIG)/pred" \
	    --metrics CLEAR HOTA Identity --threshold $(THRESHOLD) \
	    --output "$(OUTPUT_DIR)/$(TRACKER)/soccernet/$(CONFIG)/eval.json"; \
	else \
	  [ -n "$(CODABENCH_TOKEN)" ] || { echo "ERROR: CODABENCH_TOKEN not set — required for $(DATASET)."; exit 1; }; \
	  $(PYTHON) scripts/mot_format.py --dataset $(DATASET) \
	    --pred-dir "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/pred" \
	    --out-zip "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/submission.zip"; \
	  $(PYTHON) scripts/codabench_submit.py \
	    "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/submission.zip" \
	    --competition-id $(word 1,$($(DATASET)_CB)) \
	    --phase $(word 2,$($(DATASET)_CB)) \
	    --base-url "$(CODABENCH_URL)" --token "$(CODABENCH_TOKEN)" \
	    --description "$(TRACKER) $(CONFIG) — benchmark Makefile" \
	    --wait-timeout $(CODABENCH_WAIT_TIMEOUT) --poll-interval $(CODABENCH_POLL_INTERVAL) \
	    --output "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/codabench.json"; \
	fi

# Re-upload a zip without re-tracking (e.g. after Codabench daily limit resets).
upload:
	@test -f "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/submission.zip" || \
	  { echo "missing $(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/submission.zip"; exit 1; }
	@[ -n "$(CODABENCH_TOKEN)" ] || { echo "ERROR: CODABENCH_TOKEN not set."; exit 1; }
	@test "$(DATASET)" != "soccernet" || { echo "soccernet is scored locally — use track-default/track-tuned"; exit 1; }
	$(PYTHON) scripts/codabench_submit.py \
	  "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/submission.zip" \
	  --competition-id $(word 1,$($(DATASET)_CB)) \
	  --phase $(word 2,$($(DATASET)_CB)) \
	  --base-url "$(CODABENCH_URL)" --token "$(CODABENCH_TOKEN)" \
	  --description "$(TRACKER) $(CONFIG) — benchmark Makefile" \
	  --wait-timeout $(CODABENCH_WAIT_TIMEOUT) --poll-interval $(CODABENCH_POLL_INTERVAL) \
	  --output "$(OUTPUT_DIR)/$(TRACKER)/$(DATASET)/$(CONFIG)/codabench.json"

collect:
	$(PYTHON) scripts/collect.py --tracker $(TRACKER) --output-dir "$(OUTPUT_DIR)"

# Full pipeline. BENCHMARK_CONFIG=default|tuned|all (default: default only — one Codabench pass).
benchmark:
	@case "$(BENCHMARK_CONFIG)" in default|tuned|all) ;; \
	  *) echo "Set BENCHMARK_CONFIG=default, tuned, or all (or use benchmark-default / benchmark-tuned)"; exit 1;; \
	esac
	@if [ -z "$(CODABENCH_TOKEN)" ]; then \
	  echo "ERROR: CODABENCH_TOKEN must be set (required for mot17, sportsmot, dancetrack)."; exit 1; \
	fi
	@$(MAKE) prep-all
	@for d in $(DATASETS); do \
	  if [ "$(BENCHMARK_CONFIG)" = "default" ] || [ "$(BENCHMARK_CONFIG)" = "all" ]; then \
	    echo ""; echo "===== [$$d] default ====="; \
	    $(MAKE) track-default TRACKER=$(TRACKER) DATASET=$$d || exit 1; \
	  fi; \
	  if [ "$(BENCHMARK_CONFIG)" = "tuned" ] || [ "$(BENCHMARK_CONFIG)" = "all" ]; then \
	    echo ""; echo "===== [$$d] tune ====="; \
	    $(MAKE) tune TRACKER=$(TRACKER) DATASET=$$d N_TRIALS=$(N_TRIALS) || exit 1; \
	    echo "===== [$$d] tuned ====="; \
	    $(MAKE) track-tuned TRACKER=$(TRACKER) DATASET=$$d || exit 1; \
	  fi; \
	done
	@echo ""; echo "===== collect ====="
	@$(MAKE) collect TRACKER=$(TRACKER)

benchmark-default:
	@$(MAKE) benchmark BENCHMARK_CONFIG=default

benchmark-tuned:
	@$(MAKE) benchmark BENCHMARK_CONFIG=tuned

clean:
	rm -rf "$(PREP_DIR)" "$(OUTPUT_DIR)"
