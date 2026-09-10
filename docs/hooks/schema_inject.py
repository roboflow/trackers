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
# BENCH-XREF derived-claim: every "question" string below must match its FAQ
# heading in [docs/index.md](../index.md) verbatim — Google requires FAQPage
# markup to match the visible page content. The "Which tracker should I use?"
# answer additionally mirrors that heading's *answer* text and must stay
# synced with the Default HOTA tables in
# [docs/evaluations/results.md](../evaluations/results.md) if a leader changes.
_HOMEPAGE_FAQ = [
    {
        "question": "Which tracker should I use?",
        "answer": (
            "Start with ByteTrack — it's the default, has no extra dependencies, handles "
            "variable-confidence detectors well, and runs at real time latency. For the "
            "highest accuracy, McByte leads HOTA on every benchmark at default parameters "
            "but requires optional SAM/Cutie mask dependencies; OC-SORT is the best option "
            "when camera motion is significant. Use SORT if speed or device constraints "
            "require the lightest possible tracker."
        ),
    },
    {
        "question": "What is multi-object tracking and how does it differ from object detection?",
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
            "Use trackers download --name <dataset> to pull the assets available "
            "for that dataset and split: MOT17 ships frames, annotations, and "
            "pre-computed detections (test split has no annotations); SportsMOT "
            "ships frames and annotations only, with no pre-computed detections "
            "asset (test split has frames only). DanceTrack and SoccerNet-tracking "
            "support is coming soon."
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

# Per-page TechArticle.image (keyed by page.file.src_path). The brand SVG
# represents no single page's content, so pages without a specific poster
# get no image rather than a generic one.
_ARTICLE_IMAGES = {
    "trackers/sort.md": "assets/sort-demo-poster.webp",
    "trackers/bytetrack.md": "assets/bytetrack-demo-poster.webp",
    "trackers/ocsort.md": "assets/ocsort-demo-poster.webp",
    "trackers/botsort.md": "assets/botsort-demo-poster.webp",
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

    site_url = (config.get("site_url") or "https://trackers.roboflow.com").rstrip("/")

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
                item_url = _resolve_nav_item_url(item)
                # Nav sections (Home, Usage, Tuning, Trackers, ...) are pure
                # groupings with no page behind them, so mkdocs gives them no
                # URL — recording by name (not URL) is what schema.org's
                # ListItem allows: a "name"-only entry with no "item" key.
                # The top-level "Home" grouping duplicates the seeded Home
                # crumb; skip recording it but still recurse into its children.
                record = item.title is not None and item.title != "Home"
                if record:
                    path.append({"name": item.title, "url": item_url})
                if _find_in_nav(item.children, path):
                    return True
                if record:
                    path.pop()
            elif hasattr(item, "file") and item.file and item.file.src_path == page.file.src_path:
                return True
        return False

    section_path: list[dict[str, str]] = []
    found_in_nav = _find_in_nav(nav.items, section_path)
    if not found_in_nav:
        # Page isn't reachable from nav.items at all (excluded from `nav:` or
        # generated outside it) — no breadcrumb, rather than one asserting a
        # position the page doesn't actually hold.
        return None

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
        # Attribute access matches how the template reads it back
        # (config.extra.trackers_version in docs/overrides/main.html) —
        # both resolve to the same mutable `extra` dict on MkDocs' Config.
        config.extra["trackers_version"] = _pkg_version("trackers")
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
    site_url = (config.get("site_url") or "https://trackers.roboflow.com").rstrip("/")

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

        # Per-page poster image (recommended field) — omitted rather than
        # pointed at the generic brand SVG, which represents no page's
        # content and duplicates publisher.logo.
        poster = _ARTICLE_IMAGES.get(page.file.src_path)
        if poster:
            article["image"] = f"{site_url}/{poster}"

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

        page.meta["json_ld_article"] = json.dumps(article, ensure_ascii=False, indent=2)

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
        page.meta["json_ld_faq"] = json.dumps(faq_schema, ensure_ascii=False, indent=2)

    # ── BreadcrumbList JSON-LD ──
    breadcrumbs = _build_breadcrumbs(page, config, nav)
    if breadcrumbs:
        page.meta["json_ld_breadcrumbs"] = json.dumps(breadcrumbs, ensure_ascii=False, indent=2)

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
