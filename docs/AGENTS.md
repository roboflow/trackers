# docs/AGENTS.md

Guidance for AI coding agents working under `docs/` (and the doc-duplicate sites outside it: `README.md`, `.github/copilot-instructions.md`). Scoped here — not in the root `AGENTS.md` — because none of this applies to source/test work; keeping it out of the root file keeps every non-docs session from loading it.

<!--
BENCH-XREF MAP — canonical source: [evaluations/results.md](evaluations/results.md) (Default + Tuned tables per benchmark section).
Every other file below COPIES numbers out of that file by hand; nothing is templated/generated. If you change a
cell in results.md, grep this repo for `BENCH-XREF` (every duplicate site carries that token in an inline HTML
comment) and update every listed sibling. If you change a cell somewhere else first, go fix results.md too —
it's the source of truth, not just another copy.

Discover live: grep -rn BENCH-XREF docs/ README.md .github/copilot-instructions.md
(16 xref comments as of 2026-08-11: 5 in results.md, 6 in docs/trackers/*.md, 3 in docs/index.md, 1 in README.md, 1 in .github/copilot-instructions.md)

## Canonical tables ([evaluations/results.md](evaluations/results.md))
- id=mot17-default       (## MOT17 -> === "Default")
- id=sportsmot-default   (## SportsMOT -> === "Default")
- id=soccernet-default   (## SoccerNet-tracking -> === "Default")
- id=dancetrack-default  (## DanceTrack -> === "Default")
- Tuned tables (all 4 benchmarks) have NO external duplicates — only referenced from within results.md itself.

## Duplicate sites, per canonical row

SORT row (mot17/sportsmot/soccernet/dancetrack):
  -> [trackers/sort.md](trackers/sort.md)                      (Dataset|HOTA|IDF1|MOTA table, full row, first 3 benchmarks only)
  -> [index.md](index.md)                                      (Algorithms table, HOTA column only)
  -> [../README.md](../README.md)                              (Algorithms table, HOTA column only)
  -> [../.github/copilot-instructions.md](../.github/copilot-instructions.md) (Benchmark Results table, HOTA column only)

ByteTrack row (mot17/sportsmot/soccernet/dancetrack):
  -> [trackers/bytetrack.md](trackers/bytetrack.md)            (table, full row, first 3 benchmarks only)
  -> [index.md](index.md)                                      (L13 headline sentence: MOT17 HOTA only; Algorithms table, HOTA column)
  -> [../README.md](../README.md)                              (Algorithms table, HOTA column only)
  -> [../.github/copilot-instructions.md](../.github/copilot-instructions.md) (Benchmark Results table, HOTA column only)

OC-SORT row (mot17/sportsmot/soccernet/dancetrack):
  -> [trackers/ocsort.md](trackers/ocsort.md)                  (table, full row, first 3 benchmarks only)
  -> [index.md](index.md)                                      (L13 headline sentence: MOT17 HOTA only; Algorithms table, HOTA column)
  -> [../README.md](../README.md)                              (Algorithms table, HOTA column only)
  -> [../.github/copilot-instructions.md](../.github/copilot-instructions.md) (Benchmark Results table, HOTA column only)

BoT-SORT row (mot17/sportsmot/soccernet/dancetrack):
  -> [trackers/botsort.md](trackers/botsort.md)                (table, full row, first 3 benchmarks only)
  -> [index.md](index.md)                                      (Algorithms table, HOTA column)
  -> [../README.md](../README.md)                              (Algorithms table, HOTA column only)
  -> [../.github/copilot-instructions.md](../.github/copilot-instructions.md) (Benchmark Results table, HOTA column only)
  -> [trackers/mcbyte.md](trackers/mcbyte.md)                  (BoT-SORT baseline row, matching benchmark tab, full row)

C-BIoU row (mot17/sportsmot/soccernet/dancetrack — the only tracker doc with a DanceTrack row):
  -> [trackers/cbiou.md](trackers/cbiou.md)                    (table, full row, all 4 benchmarks)
  -> [../README.md](../README.md)                              (Algorithms table, HOTA column only)
  -> docs/index.md does NOT have a C-BIoU row — don't add one when editing, that's existing scope, not a gap.

McByte row (mot17/sportsmot/soccernet/dancetrack):
  -> [trackers/mcbyte.md](trackers/mcbyte.md)                  (McByte row, matching benchmark tab, full row)
  -> [index.md](index.md) FAQ "Which tracker should I use?" answer: "McByte leads every benchmark in our evaluation"
     — this claim is TRUE only while McByte is bolded-best in all 4 Default tables. Re-verify, don't assume.

## Structural asymmetries (intentional — do not "fix" by adding rows)
- docs/trackers/{sort,bytetrack,ocsort,botsort}.md tables cover MOT17/SportsMOT/SoccerNet only, no DanceTrack row.
- docs/trackers/cbiou.md is the only individual-tracker doc with a DanceTrack row.
- docs/index.md Algorithms table has 4 tracker rows (SORT/ByteTrack/OC-SORT/BoT-SORT), no C-BIoU/McByte rows.
- README.md Algorithms table has 5 tracker rows (adds C-BIoU vs index.md), still no McByte row (McByte covered
  in prose below the table instead, linking to [trackers/mcbyte.md](trackers/mcbyte.md)).
- .github/copilot-instructions.md Benchmark Results table has 4 tracker rows (SORT/ByteTrack/OC-SORT/BoT-SORT)
  across all 4 benchmarks, with no C-BIoU/McByte rows.
- docs/trackers/mcbyte.md reports McByte vs a BoT-SORT baseline only (not vs SORT/ByteTrack/OC-SORT/C-BIoU).

## Derived prose claims (not raw copies, but stale if the tables move)
- [index.md](index.md) FAQ: "McByte leads every benchmark in our evaluation" (needs McByte = best in all 4 Default tables)
- [evaluations/results.md](evaluations/results.md) "When to Use Each Tracker" section: multiple leader/ranking claims
  ("C-BIoU ... leads on SoccerNet when tuned", "McByte improves HOTA and IDF1 on all four datasets", etc.)
  sourced from the Default+Tuned tables above it on the same page.

## Version / methodology note (not a number, but same class of drift risk)
[evaluations/results.md](evaluations/results.md) states "Results use trackers vX.Y.Z" (currently v2.6.0) — must be bumped whenever
a tracker whose results appear in the tables changes default behavior in a way that would shift these numbers
(e.g. the v2.6.0 lost-track `<` -> `<=` boundary change). Check [../CHANGELOG.md](../CHANGELOG.md) before assuming the version string
is still accurate.

## Directory note
This map was written when the page lived at docs/benchmarking/results.md; it was renamed to
[evaluations/results.md](evaluations/results.md) on 2026-08-11 to match the mkdocs nav tab name ("Evaluations"),
mirroring how docs/trackers/ already matches the "Trackers" tab. mkdocs.yml nav + redirect_maps and
docs/hooks/schema_inject.py's hardcoded `src_path == "evaluations/results.md"` check were updated in the same pass —
check both if this page ever moves again.

## Verification note for whoever wrote/updated this map
Traced every entry above directly against results.md line content (not from memory or the
prior audit report) on 2026-08-10, then cross-checked design with a stronger-model advisor pass before writing
the BENCH-XREF comments. README.md was NOT part of the original audit scope and was found to have one stale
SORT SportsMOT HOTA value (corrected to 70.8) only because this crossref exercise forced a check outside docs/.
On 2026-08-11, expanding discovery to .github/copilot-instructions.md found the same stale copied cell there —
a reminder that this map is only as complete as the last review of its search scope.
Path references above use markdown-link syntax `[label](path)` even though this whole block is an HTML comment
(invisible in rendered docs either way) — several editors still resolve links for click-through inside comments,
and it keeps the format consistent with the BENCH-XREF comments in the actual doc files.
Moved here from the root `AGENTS.md` on 2026-08-11 — scoped file, not root file, since it's docs-only guidance
(see this file's header). Paths above are relative to `docs/` (this file's directory), not repo root; the
root-AGENTS.md version used repo-root-relative paths — don't copy-paste those without adjusting.
-->
