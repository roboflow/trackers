# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MkDocs hook: inject TechArticle + speakable JSON-LD per page.

Reads page.meta.description and page.title to build a TechArticle schema object,
then stores the serialized JSON string in page.meta['json_ld_article'] so that
docs/overrides/main.html can emit it inside <script type="application/ld+json">.

Also injects:
- FAQPage JSON-LD on the homepage (index.md)
- BreadcrumbList JSON-LD for pages with navigation ancestors
- Dataset JSON-LD on the comparison page
- Citation (ScholarlyArticle) on algorithm pages
"""

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# Canonical Roboflow organization @id — shared across all Roboflow properties.
# Must match the Organization @id in docs/overrides/main.html.
ORG_ID = "https://roboflow.com/#organization"

# Fixed FAQ entries for the homepage FAQPage schema.
# NOTE: dataset list must stay in sync with trackers/datasets/manifest.py.
# Currently only MOT17 and SportsMOT are downloadable; DanceTrack and SoccerNet
# are "coming soon" (see docs/learn/download.md).
_HOMEPAGE_FAQ = [
    # BENCH-XREF derived-claim: mirrors the "Which tracker should I use?" answer in
    # [docs/index.md](../index.md). Google requires FAQPage markup to match the visible
    # answer, so update both together — and re-verify against the Default HOTA tables in
    # [docs/evaluations/results.md](../evaluations/results.md) if a leader changes.
    {
        "question": "Which tracker should I use?",
        "answer": (
            "Start with ByteTrack — it's the default, has no extra dependencies, handles "
            "variable-confidence detectors well, and runs at real time latency. For the "
            "highest accuracy, McByte leads HOTA on every benchmark at default parameters "
            "but requires optional SAM/Cutie mask dependencies; BoT-SORT is the best "
            "lightweight option when camera motion is significant. Use SORT if speed or "
            "device constraints require the lightest possible tracker."
        ),
    },
    {
        "question": "What is multi-object tracking?",
        "answer": (
            "Multi-object tracking assigns a persistent ID to each detected object across "
            "video frames, maintaining continuity through occlusions, re-entries, and "
            "camera motion. Trackers use a detect-then-track approach: a detector runs on "
            "each frame, and the tracker links detections across time using motion models "
            "and spatial matching."
        ),
    },
    {
        "question": "Do I need a specific detector?",
        "answer": (
            "No. Roboflow Trackers works with any detector that outputs "
            "supervision.Detections objects. The library ships example pipelines using "
            "RF-DETR but is compatible with YOLO, Detectron2, and any custom model."
        ),
    },
    {
        "question": "How do I evaluate my tracker?",
        "answer": (
            "Run trackers eval against a directory of ground-truth MOT-format text files. "
            "The evaluation pipeline computes HOTA, IDF1, and MOTA and prints a "
            "per-sequence and combined score table."
        ),
    },
    {
        "question": "What MOT datasets does the library support?",
        "answer": (
            "MOT17 and SportsMOT are supported for download and evaluation. "
            "Use trackers download <dataset> to pull frames, annotations, and "
            "pre-computed detections. DanceTrack and SoccerNet-tracking support "
            "is coming soon."
        ),
    },
]

# Academic citations for algorithm pages (keyed by page.file.src_path).
_CITATIONS = {
    "trackers/sort.md": {
        "name": "SORT: A Simple, Online and Realtime Tracking",
        "url": "https://arxiv.org/abs/1602.00763",
        "author": "Alex Bewley et al.",
    },
    "trackers/bytetrack.md": {
        "name": "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
        "url": "https://arxiv.org/abs/2110.06864",
        "author": "Zhang et al.",
    },
    "trackers/ocsort.md": {
        "name": "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking",
        "url": "https://arxiv.org/abs/2203.14360",
        "author": "Cao et al.",
    },
}

# Benchmark datasets shown on the comparison page.
_BENCHMARK_DATASETS = [
    {
        "name": "MOT17",
        "url": "https://motchallenge.net/data/MOT17/",
        "citation": "https://arxiv.org/abs/1603.00831",
    },
    {
        "name": "SportsMOT",
        "url": "https://deeperaction.github.io/sportsmot/",
        "citation": "https://arxiv.org/abs/2304.05170",
    },
    {
        "name": "SoccerNet-tracking",
        "url": "https://www.soccer-net.org/tasks/tracking",
        "citation": "https://arxiv.org/abs/2204.12438",
    },
    {
        "name": "DanceTrack",
        "url": "https://dancetrack.github.io/",
        "citation": "https://arxiv.org/abs/2111.14690",
    },
]


def _build_breadcrumbs(page, config, nav):  # type: ignore[no-untyped-def]
    """Build BreadcrumbList JSON-LD from navigation hierarchy.

    Returns None if the page is at the root level (no meaningful breadcrumb) or if the page is the homepage (to avoid
    "Home > Home > ..." duplication).
    """
    # Skip breadcrumbs for the homepage to avoid "Home > Home > ..." duplication.
    if page.file.src_path == "index.md":
        return None

    site_url = config.get("site_url", "https://trackers.roboflow.com").rstrip("/")

    # Walk the nav tree to find the path of sections leading to this page.
    crumbs = [{"name": "Home", "url": site_url + "/"}]

    def _resolve_nav_item_url(item):  # type: ignore[no-untyped-def]
        """Return an absolute URL for a nav item when it maps to a concrete page."""
        raw_url = getattr(item, "url", None)
        if not raw_url and hasattr(item, "file") and item.file:
            raw_url = getattr(item.file, "url", None)

        if not raw_url:
            return ""
        if raw_url.startswith(("http://", "https://")):
            return raw_url
        if raw_url.startswith("/"):
            return site_url + raw_url
        return site_url + "/" + raw_url.lstrip("/")

    def _find_in_nav(items, path):  # type: ignore[no-untyped-def]
        """Recursively search nav for the page, building the path of sections."""
        for item in items:
            if hasattr(item, "children") and item.children:
                path.append({"name": item.title, "url": _resolve_nav_item_url(item)})
                if _find_in_nav(item.children, path):
                    return True
                path.pop()
            elif (
                hasattr(item, "file")
                and item.file
                and item.file.src_path == page.file.src_path
            ):
                return True
        return False

    section_path: list[dict[str, str]] = []
    _find_in_nav(nav.items, section_path)

    crumbs.extend(section_path)
    crumbs.append({"name": page.title or "", "url": page.canonical_url or ""})

    items = []
    for i, crumb in enumerate(crumbs, start=1):
        items.append(
            {
                "@type": "ListItem",
                "position": i,
                "name": crumb["name"],
                **({"item": crumb["url"]} if crumb["url"] else {}),
            }
        )

    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }


def on_config(config):  # type: ignore[no-untyped-def]
    """Expose the installed trackers version to templates as extra.trackers_version."""
    try:
        config["extra"]["trackers_version"] = _pkg_version("trackers")
    except PackageNotFoundError:
        pass
    return config


def on_page_context(context, page, config, nav):  # type: ignore[no-untyped-def]
    """Build TechArticle + speakable JSON-LD for the page and store in page.meta."""
    description = (page.meta or {}).get("description", "")
    title = page.title or ""
    canonical = page.canonical_url or ""

    if page.meta is None:
        page.meta = {}  # type: ignore[assignment]

    # Derive base URL from mkdocs.yml site_url so this hook stays in sync with
    # deployment configuration and never drifts from the actual canonical base.
    site_url = config.get("site_url", "https://trackers.roboflow.com").rstrip("/")

    # ── TechArticle JSON-LD (pages with description only) ──
    if description:
        article = {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": title,
            "description": description,
            "url": canonical,
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": canonical,
            },
            "author": {
                "@type": "Organization",
                "@id": ORG_ID,
                "name": "Roboflow",
            },
            "publisher": {
                "@type": "Organization",
                "@id": ORG_ID,
                "name": "Roboflow",
                "logo": {
                    "@type": "ImageObject",
                    "url": f"{site_url}/assets/logo-trackers-violet.svg",
                },
            },
            "speakable": {
                "@type": "SpeakableSpecification",
                "cssSelector": ["h1", ".md-content p:first-of-type"],
            },
        }

        # datePublished / dateModified from git-revision-date-localized plugin.
        # Prefer the raw iso_date keys which are always YYYY-MM-DD regardless of
        # the plugin's "type" setting in mkdocs.yml (set by plugin v1.5+ in
        # on_page_markdown, which runs before hooks). Falls back to the
        # formatted string key — safe as long as mkdocs.yml keeps type: iso_date.
        date_modified = (page.meta or {}).get(
            "git_revision_date_localized_raw_iso_date",
            (page.meta or {}).get("git_revision_date_localized", ""),
        )
        date_created = (page.meta or {}).get(
            "git_creation_date_localized_raw_iso_date",
            (page.meta or {}).get("git_creation_date_localized", ""),
        )
        if date_modified:
            article["dateModified"] = date_modified
        if date_created:
            article["datePublished"] = date_created

        # Add citation for algorithm pages.
        src_path = page.file.src_path
        if src_path in _CITATIONS:
            cite = _CITATIONS[src_path]
            article["citation"] = {
                "@type": "ScholarlyArticle",
                "name": cite["name"],
                "url": cite["url"],
                "author": {"@type": "Person", "name": cite["author"]},
            }

        page.meta["json_ld_article"] = json.dumps(
            article, ensure_ascii=False, indent=2
        )

    # ── FAQPage JSON-LD (homepage only) ──
    if page.file.src_path == "index.md":
        faq_schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": entry["question"],
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": entry["answer"],
                    },
                }
                for entry in _HOMEPAGE_FAQ
            ],
        }
        page.meta["json_ld_faq"] = json.dumps(
            faq_schema, ensure_ascii=False, indent=2
        )

    # ── BreadcrumbList JSON-LD ──
    breadcrumbs = _build_breadcrumbs(page, config, nav)
    if breadcrumbs:
        page.meta["json_ld_breadcrumbs"] = json.dumps(
            breadcrumbs, ensure_ascii=False, indent=2
        )

    # ── Dataset JSON-LD (evaluations results page only) ──
    if page.file.src_path == "evaluations/results.md":
        datasets = []
        for ds in _BENCHMARK_DATASETS:
            datasets.append(
                json.dumps(
                    {
                        "@context": "https://schema.org",
                        "@type": "Dataset",
                        "name": ds["name"],
                        "url": ds["url"],
                        "citation": ds["citation"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        page.meta["json_ld_datasets"] = datasets

    return context
