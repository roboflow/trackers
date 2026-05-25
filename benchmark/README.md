# MOT benchmark workflow

Reproduce the numbers in [`docs/trackers/comparison.md`](../docs/trackers/comparison.md) across **MOT17**, **SportsMOT**, **DanceTrack**, and **SoccerNet-tracking**. The Makefile runs tuning, tracking, Codabench submission (where required), and local evaluation, then writes doc-style tables.

Requires trackers ≥ 2.4.

```bash
cd benchmark
make setup
```

## Quick start

```bash
cd benchmark

export DATA_ROOT="/path/to/your/datasets"
export CODABENCH_TOKEN="<your-token>"       # see Codabench below

make data-check
make benchmark-default TRACKER=bytetrack
make benchmark-tuned TRACKER=bytetrack N_TRIALS=50
```

Results: `benchmark_outputs/<tracker>/tables.md` and `summary.json`.

Run **default** and **tuned** on separate days — Codabench limits submissions per phase. SoccerNet is scored locally and does not count toward that limit.

## Codabench

MOT17, SportsMOT, and DanceTrack test metrics come from [Codabench](https://www.codabench.org/). Register for each competition before uploading (approval may be required):

| Dataset    | Competition                                            |
| ---------- | ------------------------------------------------------ |
| MOT17      | [10049](https://www.codabench.org/competitions/10049/) |
| SportsMOT  | [13077](https://www.codabench.org/competitions/13077/) |
| DanceTrack | [14885](https://www.codabench.org/competitions/14885/) |

Request an API token with a one-time `curl` call (Codabench login — only the token is stored):

```bash
curl -s -X POST https://www.codabench.org/api/api-token-auth/ \
    -H "Content-Type: application/json" \
    -d '{"username":"YOUR_USER","password":"YOUR_PASS"}'

export CODABENCH_TOKEN="<your-token>"
```

Treat `CODABENCH_TOKEN` as a secret — do not publish it. See [Codabench API docs](https://www.codabench.org/api/docs/) if the request fails.

If tracking finished but upload failed (daily limit or pending approval), re-submit the zip without re-running track:

```bash
make upload TRACKER=bytetrack DATASET=mot17 CONFIG=tuned CODABENCH_TOKEN=...
```

Then `make collect TRACKER=bytetrack` to refresh the table.

## Data setup

Point `DATA_ROOT` at the folder that directly contains `mot17/`, `sportsmot/`, etc. Default: `./data`.

```
$DATA_ROOT/
  mot17/MOT17_yolox_dets/{val,test}/...
  mot17/TrackEval/data/gt/MOT17_yolox_val/train_val/...
  mot17/{val,test}/<seq>/img1/...              # BoT-SORT CMC only
  sportsmot/sportsmot_yolox_dets/{val,test}/...
  sportsmot/TrackEval/data/gt/sportsmot/val/...
  dancetrack/dancetrack_yolox_dets/{train,val,test}/...
  dancetrack/TrackEval/data/gt/dancetrack/{train,val}/...
  dancetrack/{train,val,test}_images/...       # BoT-SORT CMC (test optional)
  soccernet/SoccerNet_dets/...
  soccernet/TrackEval/data/gt/SoccerNet_tracking/...
  soccernet/soccernet_data/tracking/{train,test}/...
```

| Source             | Assets                                                                                                                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MOT17              | `trackers download mot17`; YOLOX dets replicated locally using the [ByteTrack](https://github.com/ifzhang/ByteTrack/tree/main#data-preparation) detector setup (not their pre-packaged det zips) |
| SportsMOT          | `trackers download sportsmot`; YOLOX dets replicated locally using the [SportsMOT](https://github.com/MCG-NJU/SportsMOT) detector setup                                                          |
| DanceTrack         | [DanceTrack](https://github.com/DanceTrack/DanceTrack) / [OC-SORT dets](https://github.com/noahcao/OC_SORT)                                                                                      |
| SoccerNet-tracking | [soccer-net.org](https://www.soccer-net.org/data) (2022 tracking)                                                                                                                                |

MOT17 and SportsMOT use model detections produced in-house with YOLOX, following each benchmark’s published detector configuration — the same approach described in [`docs/trackers/comparison.md`](../docs/trackers/comparison.md#detections).

```bash
make data-check DATA_ROOT="/path/to/datasets"
```

## Splits and scoring

| Dataset            | Tune  | Score | Scoring                 |
| ------------------ | ----- | ----- | ----------------------- |
| MOT17              | val   | test  | Codabench               |
| SportsMOT          | val   | test  | Codabench               |
| DanceTrack         | train | test  | Codabench               |
| SoccerNet-tracking | train | test  | Local (`trackers eval`) |

## Commands

Run from `benchmark/`. Pass variables on the command line or export them first (`DATA_ROOT`, `CODABENCH_TOKEN`, …).

| Target              | Description                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `setup`             | Install `trackers[tune]` from the repo root                                                |
| `data-check`        | Print present/missing assets under `DATA_ROOT`                                             |
| `prep`              | Prep one dataset (`DATASET=…`) into `benchmark_prep/`                                      |
| `prep-all`          | Prep all four datasets                                                                     |
| `tune`              | Optuna search → `best_params.json` (`TRACKER=`, `DATASET=`, `N_TRIALS=`)                   |
| `track-default`     | Track test split with registry defaults, then score (`TRACKER=`, `DATASET=`)               |
| `track-tuned`       | Track test split with `best_params.json`, then score (`TRACKER=`, `DATASET=`)              |
| `upload`            | Upload an existing `submission.zip` (`TRACKER=`, `DATASET=`, `CONFIG=default` or `tuned`)  |
| `benchmark-default` | `prep-all` → track-default on all datasets → `collect`                                     |
| `benchmark-tuned`   | `prep-all` → tune + track-tuned on all datasets → `collect`                                |
| `benchmark`         | Full pipeline; set `BENCHMARK_CONFIG` to `default`, `tuned`, or `all` (default: `default`) |
| `collect`           | Rebuild `tables.md` from existing score JSONs (`TRACKER=`)                                 |
| `clean`             | Remove `benchmark_prep/` and `benchmark_outputs/`                                          |

## Usage

Full pipeline (runs `prep-all`, then `collect`):

```bash
make benchmark-default TRACKER=bytetrack CODABENCH_TOKEN=...

# Tune + tuned params (another 3 Codabench uploads + SoccerNet)
make benchmark-tuned TRACKER=bytetrack N_TRIALS=50 CODABENCH_TOKEN=...

# Skip datasets (e.g. MOT17 out of Codabench submissions for today)
make benchmark-tuned TRACKER=bytetrack N_TRIALS=5 \
    DATASETS="sportsmot dancetrack soccernet" CODABENCH_TOKEN=...

# Both passes in one command (may hit daily limits)
make benchmark BENCHMARK_CONFIG=all TRACKER=bytetrack CODABENCH_TOKEN=...
```

Skip datasets (partial run or resume):

```bash
make benchmark-tuned TRACKER=bytetrack DATASETS="sportsmot soccernet" CODABENCH_TOKEN=...
```

Single dataset or step:

```bash
make prep DATASET=mot17
make tune TRACKER=bytetrack DATASET=mot17 N_TRIALS=50
make track-default TRACKER=bytetrack DATASET=mot17
make track-tuned   TRACKER=bytetrack DATASET=mot17
make upload        TRACKER=bytetrack DATASET=mot17 CONFIG=tuned
make collect       TRACKER=bytetrack
make clean
```

### Variables

| Variable           | Default               | Purpose                                     |
| ------------------ | --------------------- | ------------------------------------------- |
| `TRACKER`          | `sort`                | `sort`, `bytetrack`, `ocsort`, `botsort`, … |
| `DATA_ROOT`        | `./data`              | Raw dataset tree                            |
| `DATASET`          | `mot17`               | Single-dataset targets                      |
| `DATASETS`         | all four              | Space-separated subset for `benchmark*`     |
| `BENCHMARK_CONFIG` | `default`             | `benchmark`: `default`, `tuned`, or `all`   |
| `CONFIG`           | —                     | `upload`: `default` or `tuned`              |
| `N_TRIALS`         | `10`                  | Optuna trials per dataset                   |
| `CODABENCH_TOKEN`  | —                     | Required for Codabench datasets             |
| `PREP_DIR`         | `./benchmark_prep`    | Prepared flat MOT dets/GT                   |
| `OUTPUT_DIR`       | `./benchmark_outputs` | Params, preds, scores, tables               |

BoT-SORT sets `FIXED_PARAMS={"enable_cmc": true}` and uses frame directories when present.

## Notes

- **Tracking bypasses `trackers track`.** `scripts/track_split.py` loads the registry directly (workaround for a shared CLI parameter bug; see issue/PR). ByteTrack/SORT/OC-SORT never receive `--images-dir` during tune.
- **MOT17 server format.** `scripts/mot_format.py` triplicates `MOT17-XX.txt` into FRCNN/SDP/DPM files and stubs missing sequences for Codabench.
- **Resuming.** Steps are independent. Re-run `collect` after late uploads; use `upload` to submit an existing zip without re-tracking.
- Paths and splits live in `scripts/datasets.py`.
