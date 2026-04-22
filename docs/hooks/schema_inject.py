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
"""

import json

# Canonical Roboflow organization @id — shared across all Roboflow properties.
# Must match the Organization @id in docs/overrides/main.html.
ORG_ID = "https://roboflow.com/#organization"

# Fixed FAQ entries for the homepage FAQPage schema.
_HOMEPAGE_FAQ = [
    {
        "question": "Which tracker should I use?",
        "answer": (
            "Start with ByteTrack — it performs best across two out of four benchmarks "
            "and handles variable-confidence detectors well. Use SORT if speed or device "
            "constraints require the lightest possible tracker. Use OC-SORT when camera "
            "motion is significant or objects follow non-linear paths."
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
            "MOT17, MOT20, SportsMOT, SoccerNet-tracking, and DanceTrack are supported "
            "for download and evaluation. Use trackers download <dataset> to pull frames, "
            "annotations, and pre-computed detections."
        ),
    },
]


def _build_breadcrumbs(page, config, nav):
    """Build BreadcrumbList JSON-LD from navigation hierarchy.

    Returns None if the page is at the root level (no meaningful breadcrumb).
    """
    site_url = config.get("site_url", "https://trackers.roboflow.com").rstrip("/")

    # Walk the nav tree to find the path of sections leading to this page.
    crumbs = [{"name": "Home", "url": site_url + "/"}]

    def _find_in_nav(items, path):
        """Recursively search nav for the page, building the path of sections."""
        for item in items:
            if hasattr(item, "children") and item.children:
                path.append({"name": item.title, "url": ""})
                if _find_in_nav(item.children, path):
                    return True
                path.pop()
            elif hasattr(item, "file") and item.file and item.file.src_path == page.file.src_path:
                return True
        return False

    section_path = []
    _find_in_nav(nav.items, section_path)

    if not section_path:
        return None

    crumbs.extend(section_path)
    crumbs.append({"name": page.title or "", "url": page.canonical_url or ""})

    items = []
    for i, crumb in enumerate(crumbs, start=1):
        items.append({
            "@type": "ListItem",
            "position": i,
            "name": crumb["name"],
            **({"item": crumb["url"]} if crumb["url"] else {}),
        })

    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }


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
            "image": {
                "@type": "ImageObject",
                "url": f"{site_url}/assets/logo-trackers-violet.svg",
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
        # The plugin sets page.meta keys before hooks run.
        date_modified = (page.meta or {}).get("git_revision_date_localized", "")
        date_created = (page.meta or {}).get("git_creation_date_localized", "")
        if date_modified:
            article["dateModified"] = date_modified
        if date_created:
            article["datePublished"] = date_created

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

    return context
