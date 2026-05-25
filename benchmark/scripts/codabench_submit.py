#!/usr/bin/env python3
"""Upload a submission zip to a Codabench competition phase.

Uses Codabench's REST API (token auth + 3-step file upload). See:
  https://www.codabench.org/api/docs/
  https://docs.codabench.org/v1.23/Developers_and_Administrators/Robot-submissions/
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_METRICS = ("HOTA", "IDF1", "MOTA")
TERMINAL_STATUSES = {"finished", "failed", "cancelled", "none"}

# Known test-server presets (Makefile sets these via CODABENCH_* env vars):
#   mot17:      competition 10049, phase 16382
#   sportsmot:  competition 13077, phase 21402
#   dancetrack: competition 14885, phase 24635
DEFAULT_COMPETITION_ID = 10049
DEFAULT_PHASE_ID = 16382


def _request(
    *,
    method: str,
    url: str,
    token: str | None = None,
    data: bytes | None = None,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    hdrs = dict(headers or {})
    if token:
        hdrs["Authorization"] = f"Token {token}"
    body: bytes | None = data
    if json_body is not None:
        body = json.dumps(json_body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
        detail = raw.decode(errors="replace")
        if detail.lstrip().startswith("<!DOCTYPE") or detail.lstrip().startswith("<html"):
            detail = f"{detail[:200].strip()}... (HTML error page — check API URL/auth)"
        raise RuntimeError(f"{method} {url} → HTTP {status}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc

    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode())
    except json.JSONDecodeError:
        return status, raw.decode(errors="replace")


def _put_presigned_url(url: str, data: bytes, *, content_type: str = "application/zip") -> None:
    """PUT bytes to a presigned MinIO/S3 URL without re-encoding the query string.

    urllib.request re-quotes presigned URLs and breaks AWS signatures (403
    SignatureDoesNotMatch). Send path?query verbatim via http.client instead.
    """
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"Unsupported presigned URL scheme: {parsed.scheme!r}")

    path = parsed.path
    if parsed.query:
        path = f"{path}?{parsed.query}"

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(data)),
    }
    conn_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_class(parsed.netloc, timeout=120)
    try:
        conn.request("PUT", path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status >= 400:
            detail = raw.decode(errors="replace")
            raise RuntimeError(
                f"PUT presigned upload → HTTP {resp.status}: {detail}"
            )
    finally:
        conn.close()


def fetch_token(base_url: str, username: str, password: str) -> str:
    _, payload = _request(
        method="POST",
        url=f"{base_url.rstrip('/')}/api/api-token-auth/",
        json_body={"username": username, "password": password},
    )
    if not isinstance(payload, dict) or "token" not in payload:
        raise RuntimeError(f"Unexpected token response: {payload!r}")
    return str(payload["token"])


def can_make_submission(base_url: str, token: str, phase_id: int) -> tuple[bool, str]:
    _, payload = _request(
        method="GET",
        url=f"{base_url.rstrip('/')}/api/can_make_submission/{phase_id}/",
        token=token,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected can_make_submission response: {payload!r}")
    return bool(payload.get("can")), str(payload.get("reason", ""))


def _dataset_name(zip_path: Path, dataset_name: str | None = None) -> str:
    """Codabench internal dataset label (must be unique per user account)."""
    if dataset_name:
        return dataset_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{zip_path.stem}_{stamp}"


def upload_submission(
    *,
    base_url: str,
    token: str,
    phase_id: int,
    zip_path: Path,
    description: str | None = None,
    dataset_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Submission zip not found: {zip_path}")

    allowed, reason = can_make_submission(base_url, token, phase_id)
    if not allowed:
        raise RuntimeError(
            f"Cannot submit to phase {phase_id}: {reason or 'unknown reason'}. "
            "Register for the competition on Codabench and wait for approval if needed."
        )

    bundle_bytes = zip_path.read_bytes()
    if dry_run:
        print(
            f"Dry run: would upload {zip_path.name} ({len(bundle_bytes)} bytes) "
            f"to phase {phase_id} on {base_url}"
        )
        return {"dry_run": True, "phase": phase_id, "zip": str(zip_path)}

    _, data_record = _request(
        method="POST",
        url=f"{base_url.rstrip('/')}/api/datasets/",
        token=token,
        json_body={
            "type": "submission",
            "file_size": len(bundle_bytes),
            "request_sassy_file_name": zip_path.name,
            "name": _dataset_name(zip_path, dataset_name),
        },
    )
    if not isinstance(data_record, dict):
        raise RuntimeError(f"Unexpected /api/datasets/ response: {data_record!r}")
    key = data_record["key"]
    sassy_url = data_record["sassy_url"]

    _put_presigned_url(sassy_url, bundle_bytes, content_type="application/zip")

    _request(
        method="PUT",
        url=f"{base_url.rstrip('/')}/api/datasets/completed/{key}/",
        token=token,
    )

    body: dict[str, Any] = {"data": key, "phase": phase_id}
    if description:
        body["description"] = description
    _, submission = _request(
        method="POST",
        url=f"{base_url.rstrip('/')}/api/submissions/",
        token=token,
        json_body=body,
    )
    if not isinstance(submission, dict):
        raise RuntimeError(f"Unexpected /api/submissions/ response: {submission!r}")
    return submission


def get_submission_details(*, base_url: str, token: str, submission_id: int) -> dict[str, Any]:
    _, payload = _request(
        method="GET",
        url=f"{base_url.rstrip('/')}/api/submissions/{submission_id}/get_details/",
        token=token,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unexpected /api/submissions/{submission_id}/get_details/ response: {payload!r}"
        )
    return payload


def print_submission_failure_logs(
    *,
    base_url: str,
    token: str,
    submission_id: int,
    max_chars: int = 4000,
) -> None:
    """Best-effort scrape of scoring logs after a failed submission."""
    try:
        details = get_submission_details(
            base_url=base_url, token=token, submission_id=submission_id
        )
    except RuntimeError as exc:
        print(f"  logs unavailable: {exc}", flush=True)
        return

    chunks: list[str] = []
    for key in ("logs", "scoring_result", "prediction_result"):
        value = details.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for field in ("name", "data_file", "url"):
                        if item.get(field):
                            chunks.append(f"{key}/{item.get('name', field)}: {item[field]}")
                elif item:
                    chunks.append(str(item))
        elif isinstance(value, str) and value.strip():
            chunks.append(value)

    if not chunks:
        print("  logs: (none returned by API — open submission on Codabench for full logs)", flush=True)
        return

    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[-max_chars:]
        text = f"...(truncated)\n{text}"
    print(f"  logs →\n{text}", flush=True)


def get_submission(*, base_url: str, token: str, submission_id: int) -> dict[str, Any]:
    _, payload = _request(
        method="GET",
        url=f"{base_url.rstrip('/')}/api/submissions/{submission_id}/",
        token=token,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected /api/submissions/{submission_id}/ response: {payload!r}")
    return payload


def extract_metric_scores(
    submission: dict[str, Any],
    metric_keys: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Pull leaderboard scores by column_key (case-insensitive)."""
    wanted = {k.lower(): k for k in metric_keys}
    found: dict[str, float] = {}

    def _collect(scores: Any) -> None:
        if not isinstance(scores, list):
            return
        for item in scores:
            if not isinstance(item, dict):
                continue
            raw_key = str(item.get("column_key", ""))
            canonical = wanted.get(raw_key.lower())
            if canonical is None:
                continue
            score = item.get("score")
            if score is None:
                continue
            found[canonical] = float(score)

    _collect(submission.get("scores"))
    for child in submission.get("children") or []:
        if isinstance(child, dict):
            _collect(child.get("scores"))

    return found


def poll_submission(
    *,
    base_url: str,
    token: str,
    submission_id: int,
    timeout_seconds: float = 3600.0,
    interval_seconds: float = 10.0,
    metric_keys: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Poll GET /api/submissions/<id>/ until Finished/Failed/Cancelled."""
    start = time.monotonic()
    wait = interval_seconds
    max_wait = interval_seconds * 6
    last_status = ""

    while True:
        submission = get_submission(
            base_url=base_url, token=token, submission_id=submission_id
        )
        status = str(submission.get("status", ""))
        status_lc = status.lower()

        if status != last_status:
            print(f"  submission {submission_id}: {status}", flush=True)
            last_status = status

        if status_lc in TERMINAL_STATUSES:
            if status_lc == "finished":
                scores = extract_metric_scores(submission, metric_keys)
                if scores:
                    parts = ", ".join(f"{k}={scores[k]:.3f}" for k in metric_keys if k in scores)
                    print(f"  scores → {parts}")
                elif submission.get("scores"):
                    print("  scores → (present but no matching HOTA/IDF1/MOTA column keys)")
                else:
                    print("  scores → not available yet (check competition page)")
            elif submission.get("status_details"):
                print(f"  details → {submission['status_details']}")
            if status_lc == "failed":
                print_submission_failure_logs(
                    base_url=base_url, token=token, submission_id=submission_id
                )
            return submission

        elapsed = time.monotonic() - start
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise RuntimeError(
                f"Timed out after {timeout_seconds:.0f}s waiting for submission "
                f"{submission_id} (last status: {status})"
            )

        time.sleep(min(wait, remaining))
        wait = min(wait * 1.5, max_wait)


def resolve_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    token = os.environ.get("CODABENCH_TOKEN", "").strip()
    if token:
        return token
    username = args.username or os.environ.get("CODABENCH_USERNAME", "").strip()
    password = args.password or os.environ.get("CODABENCH_PASSWORD", "").strip()
    if username and password:
        return fetch_token(args.base_url, username, password)
    raise RuntimeError(
        "Missing API token. Set CODABENCH_TOKEN or pass --token, "
        "or set CODABENCH_USERNAME and CODABENCH_PASSWORD. "
        "Create a token via POST /api/api-token-auth/ (see Codabench API docs)."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "zip_path",
        nargs="?",
        type=Path,
        help="Submission .zip (flat MOT txt files at archive root). Omit with --submission-id.",
    )
    p.add_argument(
        "--submission-id",
        type=int,
        help="Poll an existing submission instead of uploading (use with --wait).",
    )
    p.add_argument(
        "--phase",
        type=int,
        default=int(os.environ.get("CODABENCH_PHASE", str(DEFAULT_PHASE_ID))),
        help="Codabench phase id (mot17: 16382, sportsmot: 21402).",
    )
    p.add_argument(
        "--competition-id",
        type=int,
        default=int(os.environ.get("CODABENCH_COMPETITION", str(DEFAULT_COMPETITION_ID))),
        help="Codabench competition id for result URL (mot17: 10049, sportsmot: 13077).",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("CODABENCH_URL", "https://www.codabench.org"),
        help="Codabench base URL.",
    )
    p.add_argument("--token", help="API token (or env CODABENCH_TOKEN).")
    p.add_argument("--username", help="Username for token auth (or env CODABENCH_USERNAME).")
    p.add_argument("--password", help="Password for token auth (or env CODABENCH_PASSWORD).")
    p.add_argument("--description", default="", help="Optional submission description.")
    p.add_argument(
        "--dataset-name",
        help="Codabench dataset label (default: <zip_stem>_<UTC timestamp>; must be unique).",
    )
    p.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Poll until Codabench finishes scoring (default: on).",
    )
    p.add_argument(
        "--wait-timeout",
        type=float,
        default=float(os.environ.get("CODABENCH_WAIT_TIMEOUT", "3600")),
        help="Max seconds to wait for scoring (default: 3600).",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("CODABENCH_POLL_INTERVAL", "10")),
        help="Initial poll interval in seconds (default: 10, backs off).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Leaderboard columns to print when finished (default: HOTA IDF1 MOTA).",
    )
    p.add_argument("--dry-run", action="store_true", help="Check eligibility only; do not upload.")
    args = p.parse_args(argv)

    metric_keys = tuple(args.metrics)

    try:
        token = resolve_token(args)

        if args.submission_id is not None:
            sub_id = args.submission_id
            if args.zip_path is not None:
                print("Note: zip_path ignored when --submission-id is set")
            if args.wait:
                print(f"Waiting for submission {sub_id} on {args.base_url} ...")
                submission = poll_submission(
                    base_url=args.base_url,
                    token=token,
                    submission_id=sub_id,
                    timeout_seconds=args.wait_timeout,
                    interval_seconds=args.poll_interval,
                    metric_keys=metric_keys,
                )
            else:
                submission = get_submission(
                    base_url=args.base_url, token=token, submission_id=sub_id
                )
                scores = extract_metric_scores(submission, metric_keys)
                if scores:
                    parts = ", ".join(f"{k}={scores[k]:.3f}" for k in metric_keys if k in scores)
                    print(f"scores → {parts}")
        else:
            if args.zip_path is None:
                raise RuntimeError("Provide zip_path or --submission-id")
            submission = upload_submission(
                base_url=args.base_url,
                token=token,
                phase_id=args.phase,
                zip_path=args.zip_path,
                description=args.description or None,
                dataset_name=args.dataset_name or None,
                dry_run=args.dry_run,
            )
            if submission.get("dry_run"):
                return 0
            sub_id = submission.get("id")
            print(f"Submitted → id={sub_id} status={submission.get('status')}")
            if sub_id is not None and args.wait:
                print(f"Waiting for submission {sub_id} on {args.base_url} ...")
                submission = poll_submission(
                    base_url=args.base_url,
                    token=token,
                    submission_id=int(sub_id),
                    timeout_seconds=args.wait_timeout,
                    interval_seconds=args.poll_interval,
                    metric_keys=metric_keys,
                )
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sub_id = submission.get("id")
    comp_id = args.competition_id
    phase = submission.get("phase")
    if isinstance(phase, dict):
        comp_id = phase.get("competition", comp_id)

    print(f"Done → id={sub_id} status={submission.get('status')}")
    if sub_id is not None:
        print(f"  https://www.codabench.org/competitions/{comp_id}/")
    return 0 if str(submission.get("status", "")).lower() == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
