# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MkDocs hook: inject TechArticle + speakable JSON-LD per page.

Reads page.meta.description and page.title to build a TechArticle schema object,
then stores the serialized JSON string in page.meta['json_ld_article'] so that
docs/overrides/main.html can emit it inside <script type="application/ld+json">.
"""

import json

# Canonical Roboflow organization @id — shared across all Roboflow properties.
# Must match the Organization @id in docs/overrides/main.html.
ORG_ID = "https://roboflow.com/#organization"


def on_page_context(context, page, config, nav):  # type: ignore[no-untyped-def]
    """Build TechArticle + speakable JSON-LD for the page and store in page.meta."""
    description = (page.meta or {}).get("description", "")
    title = page.title or ""
    canonical = page.canonical_url or ""

    if not description:
        return context

    # Derive base URL from mkdocs.yml site_url so this hook stays in sync with
    # deployment configuration and never drifts from the actual canonical base.
    site_url = config.get("site_url", "https://trackers.roboflow.com").rstrip("/")

    article = {
        "@context": "https://schema.org",
        "@type": "TechArticle",
        "headline": title,
        "description": description,
        "url": canonical,
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

    if page.meta is None:
        page.meta = {}  # type: ignore[assignment]
    page.meta["json_ld_article"] = json.dumps(article, ensure_ascii=False, indent=2)

    return context
