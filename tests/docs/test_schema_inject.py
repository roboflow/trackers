# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

# docs/ is not on pythonpath (pyproject.toml scopes pythonpath to "src"), and
# docs/hooks/schema_inject.py is a mkdocs hook module, not a package — load it
# directly from disk instead of a normal import.
_HOOK_PATH = Path(__file__).resolve().parents[2] / "docs" / "hooks" / "schema_inject.py"
_spec = importlib.util.spec_from_file_location("schema_inject", _HOOK_PATH)
assert _spec is not None and _spec.loader is not None
schema_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(schema_inject)


class _FakeNavFile:
    """Duck-types mkdocs `File`: only `src_path`/`url` are read by the hook."""

    def __init__(self, src_path: str, url: str = "") -> None:
        self.src_path = src_path
        self.url = url


class _FakeNavPage:
    """Duck-types a leaf mkdocs nav item: has `.file`, no `.children`."""

    def __init__(self, src_path: str, url: str = "") -> None:
        self.file = _FakeNavFile(src_path, url)


class _FakeNavSection:
    """Duck-types a mkdocs nav `Section`: has `.title`/`.children`, no `.url`/`.file`."""

    def __init__(self, title: str, children: list[Any]) -> None:
        self.title = title
        self.children = children


class _FakeNav:
    """Duck-types mkdocs `Navigation`: only `.items` is read by the hook."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items


class _FakePage:
    """Duck-types the mkdocs `Page` object passed to the hook."""

    def __init__(
        self,
        src_path: str,
        title: str = "Page Title",
        canonical_url: str = "https://trackers.roboflow.com/page/",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.file = _FakeNavFile(src_path)
        self.title = title
        self.canonical_url = canonical_url
        self.meta = meta


class _FakeConfig:
    """Duck-types the subset of mkdocs `Config` the hook touches: dict-style `.get()` for `site_url` plus attribute
    access to `.extra` (mkdocs' real `Config` exposes both item and attribute access over the same data)."""

    def __init__(
        self,
        site_url: str = "https://trackers.roboflow.com",
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._data = {"site_url": site_url}
        self.extra = extra if extra is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


_SITE_CONFIG = _FakeConfig()


class TestBuildBreadcrumbs:
    """`_build_breadcrumbs` walks `nav.items` to find the page's section path."""

    def test_emits_breadcrumbs_for_top_level_page(self) -> None:
        """A page found directly in `nav.items`, with no section ancestor, still gets a Home > Page breadcrumb."""
        nav = _FakeNav(items=[_FakeNavPage("details.md")])
        page = _FakePage(
            "details.md",
            title="Details",
            canonical_url="https://trackers.roboflow.com/details/",
        )

        result = schema_inject._build_breadcrumbs(page, _SITE_CONFIG, nav)

        assert result is not None
        names = [crumb["name"] for crumb in result["itemListElement"]]
        assert names == ["Home", "Details"]
        assert result["itemListElement"][0]["item"] == "https://trackers.roboflow.com/"
        assert result["itemListElement"][1]["item"] == page.canonical_url

    def test_includes_url_less_section_crumbs_two_levels_deep(self) -> None:
        """Nested URL-less nav sections (mkdocs `Section` objects never carry a URL) are still recorded by name, as
        regressed by the `_find_in_nav` `record` condition fix."""
        nav = _FakeNav(
            items=[
                _FakeNavSection(
                    "Trackers",
                    [
                        _FakeNavSection(
                            "Advanced",
                            [_FakeNavPage("trackers/sort.md")],
                        )
                    ],
                )
            ]
        )
        page = _FakePage(
            "trackers/sort.md",
            title="SORT",
            canonical_url="https://trackers.roboflow.com/trackers/sort/",
        )

        result = schema_inject._build_breadcrumbs(page, _SITE_CONFIG, nav)

        assert result is not None
        items = result["itemListElement"]
        assert [crumb["name"] for crumb in items] == [
            "Home",
            "Trackers",
            "Advanced",
            "SORT",
        ]
        assert "item" not in items[1]
        assert "item" not in items[2]
        assert items[3]["item"] == page.canonical_url

    def test_does_not_double_record_top_level_home_section(self) -> None:
        """The top-level "Home" nav grouping is skipped in the crumb path (it would duplicate the seeded Home crumb),
        but its children are still recursed into."""
        nav = _FakeNav(
            items=[
                _FakeNavSection(
                    "Home",
                    [_FakeNavPage("index.md"), _FakeNavPage("details.md")],
                )
            ]
        )
        page = _FakePage(
            "details.md",
            title="Details",
            canonical_url="https://trackers.roboflow.com/details/",
        )

        result = schema_inject._build_breadcrumbs(page, _SITE_CONFIG, nav)

        assert result is not None
        names = [crumb["name"] for crumb in result["itemListElement"]]
        assert names == ["Home", "Details"]
        assert names.count("Home") == 1

    def test_returns_none_for_homepage(self) -> None:
        """The homepage (`index.md`) never gets a breadcrumb, avoiding a "Home > Home > ..." duplication."""
        nav = _FakeNav(items=[_FakeNavPage("index.md")])
        page = _FakePage("index.md")

        result = schema_inject._build_breadcrumbs(page, _SITE_CONFIG, nav)

        assert result is None

    def test_returns_none_when_page_absent_from_nav(self) -> None:
        """A page excluded from `nav:` (or generated outside it) gets no breadcrumb, rather than one asserting a
        position it doesn't hold."""
        nav = _FakeNav(items=[_FakeNavPage("other.md")])
        page = _FakePage("missing.md", title="Missing")

        result = schema_inject._build_breadcrumbs(page, _SITE_CONFIG, nav)

        assert result is None


class TestOnConfig:
    """`on_config` exposes the installed trackers version to templates."""

    def test_sets_trackers_version_when_package_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolvable `trackers` package version is stored in `extra`."""
        monkeypatch.setattr(schema_inject, "_pkg_version", lambda _name: "9.9.9")
        config = _FakeConfig(extra={})

        result = schema_inject.on_config(config)

        assert result.extra["trackers_version"] == "9.9.9"

    def test_leaves_extra_untouched_when_package_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `PackageNotFoundError` is swallowed and `extra` is left as-is, instead of propagating and breaking the
        mkdocs build."""

        def _raise(_name: str) -> str:
            raise schema_inject.PackageNotFoundError(_name)

        monkeypatch.setattr(schema_inject, "_pkg_version", _raise)
        config = _FakeConfig(extra={})

        result = schema_inject.on_config(config)

        assert "trackers_version" not in result.extra


class TestOnPageContext:
    """`on_page_context` builds per-page JSON-LD and stores it in `page.meta`."""

    def test_faq_questions_match_homepage_source_of_truth(self) -> None:
        """The FAQPage question strings come verbatim from `_HOMEPAGE_FAQ`, which must stay in sync with the visible FAQ
        heading text in `docs/index.md`."""
        page = _FakePage(
            "index.md", title="Home", canonical_url="https://trackers.roboflow.com/"
        )
        page.meta = {}
        nav = _FakeNav(items=[_FakeNavPage("index.md")])

        schema_inject.on_page_context({}, page, _SITE_CONFIG, nav)

        faq = json.loads(page.meta["json_ld_faq"])
        questions = [entry["name"] for entry in faq["mainEntity"]]
        expected = [entry["question"] for entry in schema_inject._HOMEPAGE_FAQ]
        assert questions == expected

    def test_sets_article_image_for_known_src_path(self) -> None:
        """A page listed in `_ARTICLE_IMAGES` gets its poster image on the TechArticle schema."""
        page = _FakePage(
            "trackers/sort.md",
            title="SORT",
            canonical_url="https://trackers.roboflow.com/trackers/sort/",
        )
        page.meta = {"description": "SORT tracker docs."}
        nav = _FakeNav(items=[_FakeNavPage("trackers/sort.md")])

        schema_inject.on_page_context({}, page, _SITE_CONFIG, nav)

        article = json.loads(page.meta["json_ld_article"])
        assert article["image"] == "https://trackers.roboflow.com/assets/sort-demo-poster.webp"

    def test_omits_article_image_for_unknown_src_path(self) -> None:
        """A page not listed in `_ARTICLE_IMAGES` gets no `image` field, rather than falling back to a generic brand
        asset."""
        page = _FakePage(
            "guides/track.md",
            title="Track Objects",
            canonical_url="https://trackers.roboflow.com/guides/track/",
        )
        page.meta = {"description": "Track objects docs."}
        nav = _FakeNav(items=[_FakeNavPage("guides/track.md")])

        schema_inject.on_page_context({}, page, _SITE_CONFIG, nav)

        article = json.loads(page.meta["json_ld_article"])
        assert "image" not in article
