#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
FILM_DIR="$ROOT_DIR/film"
CACHE_DIR="$ROOT_DIR/.cache"
MPL_CACHE_DIR="$CACHE_DIR/matplotlib"
FONTCONFIG_CACHE_DIR="$CACHE_DIR/fontconfig"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"

CLIP_START_INPUT=""
CLIP_DURATION_INPUT="60"
RANDOM_MIDDLE_MINUTE=0
CLIP_ONLY=0
RUN_NAME=""
FORWARD_ARGS=()

list_videos() {
  find "$FILM_DIR" -maxdepth 1 -type f \
    \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.avi" \) \
    | sort
}

usage() {
  echo "Usage: ./scripts/track_video.sh <video-path> [wrapper args] [extra trackers args]"
  echo
  echo "Examples:"
  echo "  ./scripts/track_video.sh \"film/match.mp4\""
  echo "  ./scripts/track_video.sh \"film/match.mp4\" --clip-start 00:25:00 --clip-duration 60"
  echo "  ./scripts/track_video.sh \"film/match.mp4\" --random-middle-minute --clip-only"
  echo "  ./scripts/track_video.sh \"film/match.mp4\" --run-name my-benchmark"
  echo "  ./scripts/track_video.sh \"film/match.mp4\" --display"
  echo "  TRACKERS_MODEL=rfdetr-nano ./scripts/track_video.sh \"film/match.mp4\""
  echo
  echo "Wrapper args:"
  echo "  --clip-start HH:MM:SS|SECONDS   Cut from a specific offset before tracking"
  echo "  --clip-duration HH:MM:SS|SECONDS Duration of the clip (default: 60)"
  echo "  --random-middle-minute          Pick a random 60s window from the middle third"
  echo "  --clip-only                     Create the clip and stop without tracking"
  echo "  --run-name NAME                Override the output folder name under runs/"
  echo
  echo "Defaults:"
  echo "  model:   \${TRACKERS_MODEL:-rfdetr-medium}"
  echo "  tracker: \${TRACKERS_TRACKER:-bytetrack}"
  echo "  classes: \${TRACKERS_CLASSES:-person}"
  echo
  echo "Outputs:"
  echo "  runs/<video-name>/tracked.mp4"
  echo "  runs/<video-name>/tracks.txt"
}

parse_seconds() {
  local input="$1"
  local hours=0
  local minutes=0
  local seconds=0
  local fractional=""

  if [[ "$input" == *:* ]]; then
    IFS=":" read -r -a parts <<< "$input"
    if [[ "${#parts[@]}" -eq 3 ]]; then
      hours="${parts[0]}"
      minutes="${parts[1]}"
      seconds="${parts[2]}"
    elif [[ "${#parts[@]}" -eq 2 ]]; then
      minutes="${parts[0]}"
      seconds="${parts[1]}"
    elif [[ "${#parts[@]}" -eq 1 ]]; then
      seconds="${parts[0]}"
    else
      echo "Invalid time value: $input" >&2
      exit 1
    fi
  else
    seconds="$input"
  fi

  fractional="${seconds#*.}"
  seconds="${seconds%%.*}"

  if [[ -n "$fractional" && "$fractional" != "$seconds" ]]; then
    :
  fi

  if ! [[ "$hours" =~ ^[0-9]+$ && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ ]]; then
    echo "Invalid time value: $input" >&2
    exit 1
  fi

  echo $((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

format_hhmmss() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

format_slug_time() {
  local total_seconds="$1"
  printf "%02d-%02d-%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

get_duration_seconds() {
  local source_path="$1"
  local duration_raw

  duration_raw="$("$FFPROBE_BIN" -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$source_path")"

  echo "${duration_raw%.*}"
}

choose_random_middle_start() {
  local source_duration="$1"
  local clip_duration="$2"
  local max_start=0
  local lower_bound=0
  local upper_bound=0
  local span=0

  if (( source_duration <= clip_duration )); then
    echo 0
    return
  fi

  max_start=$((source_duration - clip_duration))
  lower_bound=$((source_duration / 3))
  upper_bound=$((((source_duration * 2) / 3) - clip_duration))

  if (( upper_bound > max_start )); then
    upper_bound="$max_start"
  fi

  if (( upper_bound < lower_bound )); then
    echo $((max_start / 2))
    return
  fi

  span=$((upper_bound - lower_bound + 1))
  echo $((lower_bound + RANDOM % span))
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing local virtualenv at $PYTHON_BIN"
  echo "Run: uv sync --extra detection"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage
  echo
  echo "Videos currently in film/:"
  while IFS= read -r video; do
    echo "  ${video#$ROOT_DIR/}"
  done < <(list_videos)
  exit 1
fi

SOURCE_INPUT="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clip-start)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --clip-start" >&2
        exit 1
      fi
      CLIP_START_INPUT="$2"
      shift 2
      ;;
    --clip-duration)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --clip-duration" >&2
        exit 1
      fi
      CLIP_DURATION_INPUT="$2"
      shift 2
      ;;
    --random-middle-minute)
      RANDOM_MIDDLE_MINUTE=1
      shift
      ;;
    --clip-only)
      CLIP_ONLY=1
      shift
      ;;
    --run-name)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --run-name" >&2
        exit 1
      fi
      RUN_NAME="$2"
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$SOURCE_INPUT" = /* ]]; then
  SOURCE_PATH="$SOURCE_INPUT"
else
  SOURCE_PATH="$ROOT_DIR/$SOURCE_INPUT"
fi

if [[ ! -f "$SOURCE_PATH" ]]; then
  echo "Source video not found: $SOURCE_INPUT"
  exit 1
fi

SOURCE_BASENAME="$(basename "$SOURCE_PATH")"
SOURCE_STEM="${SOURCE_BASENAME%.*}"
SAFE_STEM="$(printf '%s' "$SOURCE_STEM" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
if [[ -n "$RUN_NAME" ]]; then
  SAFE_RUN_NAME="$(printf '%s' "$RUN_NAME" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
  RUN_ROOT="$ROOT_DIR/runs/$SAFE_RUN_NAME"
else
  RUN_ROOT="$ROOT_DIR/runs/$SAFE_STEM"
fi
CLIP_SOURCE_PATH=""
CLIP_START_SECS=0
CLIP_DURATION_SECS=0
SOURCE_DURATION_SECS=0

if [[ -n "$CLIP_START_INPUT" && "$RANDOM_MIDDLE_MINUTE" -eq 1 ]]; then
  echo "Choose either --clip-start or --random-middle-minute, not both." >&2
  exit 1
fi

if [[ -n "$CLIP_START_INPUT" || "$RANDOM_MIDDLE_MINUTE" -eq 1 ]]; then
  if ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
    echo "ffmpeg is required for clip extraction." >&2
    exit 1
  fi
  if ! command -v "$FFPROBE_BIN" >/dev/null 2>&1; then
    echo "ffprobe is required for clip extraction." >&2
    exit 1
  fi

  SOURCE_DURATION_SECS="$(get_duration_seconds "$SOURCE_PATH")"
  CLIP_DURATION_SECS="$(parse_seconds "$CLIP_DURATION_INPUT")"

  if (( CLIP_DURATION_SECS <= 0 )); then
    echo "Clip duration must be greater than 0." >&2
    exit 1
  fi

  if (( SOURCE_DURATION_SECS <= CLIP_DURATION_SECS )); then
    CLIP_DURATION_SECS="$SOURCE_DURATION_SECS"
    CLIP_START_SECS=0
  elif [[ -n "$CLIP_START_INPUT" ]]; then
    CLIP_START_SECS="$(parse_seconds "$CLIP_START_INPUT")"
    if (( CLIP_START_SECS < 0 )); then
      CLIP_START_SECS=0
    fi
    if (( CLIP_START_SECS + CLIP_DURATION_SECS > SOURCE_DURATION_SECS )); then
      CLIP_START_SECS=$((SOURCE_DURATION_SECS - CLIP_DURATION_SECS))
    fi
  else
    CLIP_START_SECS="$(choose_random_middle_start "$SOURCE_DURATION_SECS" "$CLIP_DURATION_SECS")"
  fi

  CLIP_TAG="clip_$(format_slug_time "$CLIP_START_SECS")_d${CLIP_DURATION_SECS}s"
  CLIP_DIR="$RUN_ROOT/clips"
  CLIP_SOURCE_PATH="$CLIP_DIR/${CLIP_TAG}.mp4"

  mkdir -p "$CLIP_DIR"

  echo "Creating clip:     ${CLIP_SOURCE_PATH#$ROOT_DIR/}"
  echo "Clip start:        $(format_hhmmss "$CLIP_START_SECS")"
  echo "Clip duration:     $(format_hhmmss "$CLIP_DURATION_SECS")"
  echo

  "$FFMPEG_BIN" -hide_banner -loglevel error \
    -ss "$(format_hhmmss "$CLIP_START_SECS")" \
    -i "$SOURCE_PATH" \
    -t "$(format_hhmmss "$CLIP_DURATION_SECS")" \
    -c:v libx264 \
    -preset veryfast \
    -crf 18 \
    -c:a aac \
    -movflags +faststart \
    -y \
    "$CLIP_SOURCE_PATH"

  SOURCE_PATH="$CLIP_SOURCE_PATH"
  RUN_DIR="$RUN_ROOT/$CLIP_TAG"
else
  RUN_DIR="$RUN_ROOT"
fi

VIDEO_OUT="$RUN_DIR/tracked.mp4"
MOT_OUT="$RUN_DIR/tracks.txt"

mkdir -p "$RUN_DIR"
mkdir -p "$MPL_CACHE_DIR" "$FONTCONFIG_CACHE_DIR"

echo "Source video:      ${SOURCE_PATH#$ROOT_DIR/}"
echo "Annotated output:  ${VIDEO_OUT#$ROOT_DIR/}"
echo "Raw MOT output:    ${MOT_OUT#$ROOT_DIR/}"
echo

export XDG_CACHE_HOME="$CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"

if (( CLIP_ONLY == 1 )); then
  if [[ -n "$CLIP_SOURCE_PATH" ]]; then
    echo "Clip ready:        ${CLIP_SOURCE_PATH#$ROOT_DIR/}"
  else
    echo "--clip-only requires --clip-start or --random-middle-minute" >&2
    exit 1
  fi
  exit 0
fi

TRACK_CMD=(
  "$PYTHON_BIN" -m trackers.scripts.__main__ track
  --source "$SOURCE_PATH"
  --output "$VIDEO_OUT"
  --mot-output "$MOT_OUT"
  --model "${TRACKERS_MODEL:-rfdetr-medium}"
  --tracker "${TRACKERS_TRACKER:-bytetrack}"
  --classes "${TRACKERS_CLASSES:-person}"
  --model.confidence "${TRACKERS_CONFIDENCE:-0.3}"
  --model.device "${TRACKERS_DEVICE:-auto}"
  --tracker.lost_track_buffer "${TRACKERS_LOST_TRACK_BUFFER:-60}"
  --show-labels
  --show-trajectories
  --overwrite
)

if (( ${#FORWARD_ARGS[@]} > 0 )); then
  TRACK_CMD+=("${FORWARD_ARGS[@]}")
fi

"${TRACK_CMD[@]}"

echo
echo "Finished."
echo "Annotated video: ${VIDEO_OUT#$ROOT_DIR/}"
echo "Raw MOT track data: ${MOT_OUT#$ROOT_DIR/}"
