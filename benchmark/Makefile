# MOT benchmark workflow (tune / eval / submit / Codabench upload).
#
# Run from this directory:
#   cd benchmark
#   make setup
#   make eval TRACKER=sort DATASET=mot17
#   make submit upload-codabench TRACKER=sort DATASET=dancetrack CODABENCH_TOKEN=...
#
# Dataset files live under benchmark/data/ (see README.md). Override with DATA_ROOT=...

# ── Tools ─────────────────────────────────────────────────────────────────────

SHELL         := /bin/bash
ROOT          := $(CURDIR)
REPO_ROOT     := $(abspath $(ROOT)/..)
PYTHON        ?= python
TRACKERS      ?= $(PYTHON) -m trackers.scripts
TRACKERS_REPO := $(REPO_ROOT)
DATA_ROOT     ?= $(ROOT)/data

# ── Knobs ─────────────────────────────────────────────────────────────────────

TRACKER      ?= sort
DATASET      ?= dancetrack
N_TRIALS     ?= 10
OBJECTIVE    ?= HOTA
THRESHOLD    ?= 0.5
METRICS      := CLEAR HOTA Identity
SEED         ?=
PARAMS       ?=
FIXED_PARAMS ?=

CODABENCH_URL              ?= https://www.codabench.org
CODABENCH_TOKEN            ?=
CODABENCH_USERNAME         ?=
CODABENCH_PASSWORD         ?=
CODABENCH_DESCRIPTION      ?=
CODABENCH_WAIT             ?= 1
CODABENCH_WAIT_TIMEOUT     ?= 3600
CODABENCH_POLL_INTERVAL    ?= 10

PREP_DIR    := $(ROOT)/benchmark_prep
OUTPUT_DIR  := $(ROOT)/benchmark_outputs
JOB_DIR     := $(OUTPUT_DIR)/$(TRACKER)/$(DATASET)
BEST_PARAMS := $(JOB_DIR)/best_params.json

ifeq ($(DATASET),soccernet)
  TUNE_SPLIT        := train
  EVAL_SPLIT        := test
  SUBMIT_SPLIT      :=
  EVAL_GT_DIR       := $(DATA_ROOT)/soccernet/TrackEval/data/gt/SoccerNet_tracking/SoccerNet_tracking_2022_all_gts
  SEQMAP_TUNE       :=
  SEQMAP_EVAL       :=
  TUNE_IMAGES_DIR   := $(DATA_ROOT)/soccernet/soccernet_data/tracking/train
  EVAL_IMAGES_DIR   := $(DATA_ROOT)/soccernet/soccernet_data/tracking/test
  SUBMIT_IMAGES_DIR :=
else ifeq ($(DATASET),dancetrack)
  TUNE_SPLIT        := train
  EVAL_SPLIT        := val
  SUBMIT_SPLIT      := test
  EVAL_GT_DIR       := $(DATA_ROOT)/dancetrack/TrackEval/data/gt/dancetrack/val
  SEQMAP_TUNE       := $(DATA_ROOT)/dancetrack/TrackEval/data/gt/dancetrack/DanceTrack-train.txt
  SEQMAP_EVAL       := $(DATA_ROOT)/dancetrack/TrackEval/data/gt/dancetrack/DanceTrack-val.txt
  TUNE_IMAGES_DIR   := $(DATA_ROOT)/dancetrack/train_images
  EVAL_IMAGES_DIR   := $(DATA_ROOT)/dancetrack/val_images
  SUBMIT_IMAGES_DIR := $(DATA_ROOT)/dancetrack/test_images
  SUBMIT_DETS_DIR   := $(DATA_ROOT)/dancetrack/dancetrack_yolox_dets/test
else ifeq ($(DATASET),sportsmot)
  TUNE_SPLIT        := val
  EVAL_SPLIT        := val
  SUBMIT_SPLIT      := test
  EVAL_GT_DIR       := $(DATA_ROOT)/sportsmot/TrackEval/data/gt/sportsmot/val
  SEQMAP_TUNE       :=
  SEQMAP_EVAL       :=
  TUNE_IMAGES_DIR   := $(DATA_ROOT)/sportsmot/val
  EVAL_IMAGES_DIR   := $(TUNE_IMAGES_DIR)
  SUBMIT_IMAGES_DIR := $(DATA_ROOT)/sportsmot/test
  SUBMIT_DETS_DIR   := $(DATA_ROOT)/sportsmot/sportsmot_yolox_dets/test
else ifeq ($(DATASET),mot17)
  TUNE_SPLIT        := val
  EVAL_SPLIT        := val
  SUBMIT_SPLIT      := test
  EVAL_GT_DIR       := $(DATA_ROOT)/mot17/TrackEval/data/gt/MOT17_yolox_val/train_val
  SEQMAP_TUNE       := $(DATA_ROOT)/mot17/TrackEval/data/gt/MOT17/MOT17-val.txt
  SEQMAP_EVAL       := $(SEQMAP_TUNE)
  TUNE_IMAGES_DIR   := $(DATA_ROOT)/mot17/val
  EVAL_IMAGES_DIR   := $(TUNE_IMAGES_DIR)
  SUBMIT_IMAGES_DIR := $(DATA_ROOT)/mot17/test
  SUBMIT_DETS_DIR   := $(DATA_ROOT)/mot17/MOT17_yolox_dets/test
else
  $(error Unknown DATASET=$(DATASET). Use: soccernet, dancetrack, sportsmot, mot17)
endif

ifeq ($(DATASET),mot17)
  CODABENCH_COMPETITION := 10049
  CODABENCH_PHASE       := 16382
else ifeq ($(DATASET),sportsmot)
  CODABENCH_COMPETITION := 13077
  CODABENCH_PHASE       := 21402
else ifeq ($(DATASET),dancetrack)
  CODABENCH_COMPETITION := 14885
  CODABENCH_PHASE       := 24635
endif

ifeq ($(TRACKER),botsort)
  ifeq ($(strip $(FIXED_PARAMS)),)
    FIXED_PARAMS := {"enable_cmc": true}
  endif
  USE_IMAGES := 1
endif

TUNE_PREP   := $(PREP_DIR)/$(DATASET)/$(TUNE_SPLIT)
EVAL_PREP   := $(PREP_DIR)/$(DATASET)/$(EVAL_SPLIT)
PRED_DIR    := $(JOB_DIR)/pred_$(EVAL_SPLIT)
EVAL_JSON   := $(JOB_DIR)/eval_$(EVAL_SPLIT).json
SUBMIT_DIR  := $(JOB_DIR)/submit_$(SUBMIT_SPLIT)
SUBMIT_ZIP  := $(JOB_DIR)/$(TRACKER)_$(DATASET)_$(SUBMIT_SPLIT)_submission.zip

define resolve_params
if [ -n "$(PARAMS)" ]; then params_file="$(PARAMS)"; \
elif [ -f "$(BEST_PARAMS)" ]; then params_file="$(BEST_PARAMS)"; \
else echo "Using $(TRACKER) default parameters"; params_file="-"; fi; \
flags=$$($(PYTHON) scripts/tracker_flags.py $(TRACKER) "$$params_file");
endef

.PHONY: help setup tune eval submit upload-codabench all

help:
	@echo "Run from: cd benchmark && make <target>"
	@echo "Targets: setup | tune | eval | submit | upload-codabench | all"
	@echo "DATA_ROOT=$(DATA_ROOT)"

setup:
	$(PYTHON) -m pip install -e "$(TRACKERS_REPO)[tune]"
	$(PYTHON) scripts/prep_benchmark.py --data-root "$(DATA_ROOT)" --dataset $(DATASET) --split all

tune: setup
	@$(PYTHON) -c "import optuna" 2>/dev/null || { echo "Optuna missing. Run: make setup"; exit 1; }
	@if [ -n "$(USE_IMAGES)" ]; then \
	  test -d "$(TUNE_IMAGES_DIR)" || { echo "Missing $(TUNE_IMAGES_DIR)"; exit 1; }; \
	fi
	@mkdir -p "$(JOB_DIR)"
	$(TRACKERS) tune \
		--tracker $(TRACKER) \
		--gt-dir "$(TUNE_PREP)/gt" \
		--detections-dir "$(TUNE_PREP)/dets" \
		--objective $(OBJECTIVE) \
		--n-trials $(N_TRIALS) \
		--metrics $(METRICS) \
		--threshold $(THRESHOLD) \
		$(if $(SEQMAP_TUNE),--seqmap "$(SEQMAP_TUNE)",) \
		$(if $(USE_IMAGES),--images-dir "$(TUNE_IMAGES_DIR)",) \
		$(if $(FIXED_PARAMS),--fixed-params '$(FIXED_PARAMS)',) \
		$(if $(SEED),--seed $(SEED),) \
		--output "$(BEST_PARAMS)"

eval:
	@test -d "$(EVAL_PREP)/dets" || { echo "Run: make setup DATASET=$(DATASET)"; exit 1; }
	@mkdir -p "$(PRED_DIR)"
	@set -euo pipefail; \
	$(resolve_params) \
	for det in "$(EVAL_PREP)/dets"/*.txt; do \
	  seq=$$(basename "$$det" .txt); \
	  echo "eval/track $$seq"; \
	  source_args=(); \
	  if [ -n "$(USE_IMAGES)" ]; then \
	    frame_seq="$$seq"; \
	    if [ "$(DATASET)" = "mot17" ] && [[ "$$seq" != *-FRCNN ]]; then frame_seq="$$seq-FRCNN"; fi; \
	    img_dir="$(EVAL_IMAGES_DIR)/$$frame_seq/img1"; \
	    test -d "$$img_dir" || { echo "Missing $$img_dir" >&2; exit 1; }; \
	    source_args=(--source "$$img_dir"); \
	  fi; \
	  $(TRACKERS) track \
	    --detections "$$det" --tracker $(TRACKER) $$flags $${source_args[@]+"$${source_args[@]}"} \
	    --mot-output "$(PRED_DIR)/$$seq.txt" --overwrite; \
	done
	$(TRACKERS) eval \
		--gt-dir "$(EVAL_GT_DIR)" --tracker-dir "$(PRED_DIR)" \
		--metrics $(METRICS) --threshold $(THRESHOLD) \
		--columns MOTA HOTA IDF1 \
		$(if $(SEQMAP_EVAL),--seqmap "$(SEQMAP_EVAL)",) \
		--output "$(EVAL_JSON)"
	@echo "Saved → $(EVAL_JSON)"

submit:
	@test -n "$(SUBMIT_SPLIT)" || { echo "No submit split for $(DATASET)"; exit 1; }
	@test -d "$(SUBMIT_DETS_DIR)" || { echo "Missing YOLOX detections: $(SUBMIT_DETS_DIR)"; exit 1; }
	@mkdir -p "$(SUBMIT_DIR)"
	@set -euo pipefail; \
	params_args=(); \
	if [ -n "$(PARAMS)" ]; then params_args=(--params "$(PARAMS)"); \
	elif [ -f "$(BEST_PARAMS)" ]; then params_args=(--params "$(BEST_PARAMS)"); fi; \
	images_args=(); \
	if [ -n "$(USE_IMAGES)" ]; then \
	  test -d "$(SUBMIT_IMAGES_DIR)" || { echo "Missing frames: $(SUBMIT_IMAGES_DIR)" >&2; exit 1; }; \
	  images_args=(--images-dir "$(SUBMIT_IMAGES_DIR)"); \
	fi; \
	$(PYTHON) scripts/submit_yolox.py \
	  --tracker $(TRACKER) --dataset $(DATASET) --split $(SUBMIT_SPLIT) \
	  --data-root "$(DATA_ROOT)" --output-dir "$(SUBMIT_DIR)" \
	  $${params_args[@]+"$${params_args[@]}"} \
	  $${images_args[@]+"$${images_args[@]}"}
	@$(PYTHON) scripts/mot_challenge_submission_format.py "$(SUBMIT_DIR)"
	@if [ "$(DATASET)" = "mot17" ]; then \
	  $(PYTHON) scripts/mot17_server_format.py "$(SUBMIT_DIR)"; \
	fi
	@rm -f "$(SUBMIT_ZIP)"
	cd "$(SUBMIT_DIR)" && zip -r "$(SUBMIT_ZIP)" .
	@echo "Created $(SUBMIT_ZIP)"

upload-codabench:
	@test "$(DATASET)" = "mot17" -o "$(DATASET)" = "sportsmot" -o "$(DATASET)" = "dancetrack" || \
	  { echo "upload-codabench supports mot17, sportsmot, dancetrack"; exit 1; }
	@test -f "$(SUBMIT_ZIP)" || $(MAKE) submit TRACKER=$(TRACKER) DATASET=$(DATASET) PARAMS="$(PARAMS)"
	@test -n "$(CODABENCH_TOKEN)" -o -n "$(CODABENCH_USERNAME)" || \
	  { echo "Set CODABENCH_TOKEN or CODABENCH_USERNAME+CODABENCH_PASSWORD"; exit 1; }
	$(PYTHON) scripts/codabench_submit.py "$(SUBMIT_ZIP)" \
	  --phase $(CODABENCH_PHASE) --competition-id $(CODABENCH_COMPETITION) \
	  --base-url "$(CODABENCH_URL)" \
	  $(if $(CODABENCH_TOKEN),--token "$(CODABENCH_TOKEN)",) \
	  $(if $(CODABENCH_USERNAME),--username "$(CODABENCH_USERNAME)",) \
	  $(if $(CODABENCH_PASSWORD),--password "$(CODABENCH_PASSWORD)",) \
	  $(if $(CODABENCH_DESCRIPTION),--description "$(CODABENCH_DESCRIPTION)",) \
	  $(if $(filter 0 false no,$(CODABENCH_WAIT)),--no-wait,) \
	  --wait-timeout $(CODABENCH_WAIT_TIMEOUT) \
	  --poll-interval $(CODABENCH_POLL_INTERVAL)

all: tune eval submit
