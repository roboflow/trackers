# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Back-compat shims for the mask stack moved out of ``trackers.core.mcbyte``.

The shims use two mechanisms, so the tests are split the same way: the ``deprecated_class`` proxies warn on use and
forward to the real class, while the abstract base classes are served by a module ``__getattr__`` that warns on import
and hands back the real class so it stays subclassable.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from typing import Any

import numpy as np
import pytest

import trackers.core.mcbyte.mask_manager as mask_manager_shim
import trackers.core.mcbyte.masks as masks_shim
from trackers.core.masks.base import (
    MaskGenerator,
    MaskOutput,
    MaskPropagator,
    TrackletSnapshot,
)
from trackers.core.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator
from trackers.core.masks.manager import MaskManager

DEPRECATION_MATCH = r"deprecated since v2\.7\.0"

FORWARDED_ABCS = [
    pytest.param("MaskGenerator", MaskGenerator, id="mask-generator"),
    pytest.param("MaskPropagator", MaskPropagator, id="mask-propagator"),
]


def _proxy(name: str) -> Any:
    """Return one proxied name from whichever shim module declares it.

    Fetched with ``getattr`` rather than an attribute expression because the
    proxies have no useful static type: to a type checker they are the empty
    placeholder classes the decorator was applied to.
    """
    module = mask_manager_shim if name == "MaskManager" else masks_shim
    return getattr(module, name)


def _build_proxied(name: str) -> Any:
    """Instantiate one proxied name with arguments its real class accepts."""
    builders = {
        "TrackletSnapshot": lambda: _proxy(name)(7, np.array([1, 2, 3, 4], dtype=np.float32)),
        "MaskOutput": lambda: _proxy(name)(masks=None, tracklet_mask_dict={}),
        "MaskManager": lambda: _proxy(name)(
            mask_generator=DummyBoxMaskGenerator(),
            mask_propagator=DummyIdentityMaskPropagator(),
        ),
    }
    return builders[name]()


PROXIED_SYMBOLS = [
    pytest.param("TrackletSnapshot", TrackletSnapshot, id="tracklet-snapshot"),
    pytest.param("MaskOutput", MaskOutput, id="mask-output"),
    pytest.param("MaskManager", MaskManager, id="mask-manager"),
]


class TestProxiedSymbolsWarnOnUseNotOnImport:
    """``MaskOutput``, ``TrackletSnapshot`` and ``MaskManager`` are pyDeprecate proxies.

    Their warning fires when the name is *used*, which is the half of the split that differs from the abstract base
    classes below. Both directions are asserted so the asymmetry cannot drift unnoticed.
    """

    @pytest.mark.parametrize(("name", "real"), PROXIED_SYMBOLS)
    def test_importing_the_name_stays_silent(self, name: str, real: type) -> None:
        """Binding the name emits nothing: a proxy does not warn at import time."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _proxy(name)

        assert [warning for warning in caught if issubclass(warning.category, FutureWarning)] == []

    @pytest.mark.parametrize(("name", "real"), PROXIED_SYMBOLS)
    def test_using_the_name_warns_and_returns_a_real_instance(self, name: str, real: type) -> None:
        """Instantiating the proxy warns and forwards construction to the real class."""
        with pytest.warns(FutureWarning, match=DEPRECATION_MATCH):
            instance = _build_proxied(name)

        assert isinstance(instance, real)
        assert type(instance) is real

    @pytest.mark.parametrize(("name", "real"), PROXIED_SYMBOLS)
    def test_isinstance_forwards_to_the_real_class(self, name: str, real: type) -> None:
        """``isinstance`` against the proxy answers for the real class."""
        with pytest.warns(FutureWarning):
            instance = _build_proxied(name)
        proxy = _proxy(name)

        assert isinstance(instance, proxy)
        assert issubclass(real, proxy)

    @pytest.mark.parametrize(("name", "real"), PROXIED_SYMBOLS)
    def test_proxy_is_not_identical_to_the_real_class(self, name: str, real: type) -> None:
        """Documented caveat: ``proxy is real`` is False, so identity checks must not be used."""
        proxy = _proxy(name)

        assert proxy is not real

    def test_warning_names_the_old_import_path(self) -> None:
        """The proxy message points at the module to fix, not only the class name."""
        with pytest.warns(FutureWarning, match=r"trackers\.core\.mcbyte\.mask_manager\.MaskManager"):
            _build_proxied("MaskManager")

    def test_proxy_built_manager_is_accepted_by_mcbyte_tracker(self) -> None:
        """A manager built through the proxy still satisfies ``McByteTracker``."""
        from trackers.core.mcbyte.tracker import McByteTracker

        with pytest.warns(FutureWarning):
            manager = _build_proxied("MaskManager")

        tracker = McByteTracker(enable_mask_manager=True, mask_manager=manager)

        assert tracker.mask_manager is manager


class TestForwardedAbstractBasesWarnOnImport:
    """``MaskGenerator`` and ``MaskPropagator`` are served by a module ``__getattr__``.

    Unlike the proxies above, these warn at *import* and hand back the real class, which is what keeps them
    subclassable.
    """

    @pytest.mark.parametrize(("name", "expected"), FORWARDED_ABCS)
    def test_importing_the_name_warns_and_returns_the_real_class(self, name: str, expected: type) -> None:
        """Importing the name warns ``FutureWarning`` and yields the class itself."""
        with pytest.warns(FutureWarning, match=DEPRECATION_MATCH):
            symbol = getattr(masks_shim, name)

        assert symbol is expected

    def test_forwarded_abstract_base_stays_subclassable(self) -> None:
        """The real ABC comes back, so custom mask backends can still subclass it."""
        with pytest.warns(FutureWarning):
            base = masks_shim.MaskGenerator

        class _CustomGenerator(base):  # type: ignore[misc, valid-type]
            def generate(self, frame: object, tracklets: object) -> None:  # type: ignore[override]
                return None

        assert issubclass(_CustomGenerator, MaskGenerator)


class TestShimSurface:
    """Both shims export exactly what 2.6.0 exported, and nothing more."""

    def test_masks_all_matches_the_new_home(self) -> None:
        """``__all__`` still lists exactly the 2.6.0 subpackage exports."""
        import trackers.core.masks as new_home

        assert sorted(masks_shim.__all__) == sorted(new_home.__all__)

    def test_mask_manager_all_lists_only_the_moved_class(self) -> None:
        """``__all__`` still lists exactly the 2.6.0 module export."""
        assert mask_manager_shim.__all__ == ["MaskManager"]

    def test_masks_rejects_a_name_it_never_exported(self) -> None:
        """A deep submodule symbol is not invented by the shim."""
        with pytest.raises(AttributeError, match="has no attribute 'CutieMaskPropagator'"):
            masks_shim.CutieMaskPropagator

    def test_mask_manager_rejects_a_name_it_never_exported(self) -> None:
        """A name that lived in the sibling subpackage is not served here."""
        with pytest.raises(AttributeError, match="has no attribute 'MaskGenerator'"):
            getattr(mask_manager_shim, "MaskGenerator")


def test_importing_trackers_does_not_warn_or_import_torch() -> None:
    """The shims stay lazy: plain ``import trackers`` neither warns nor pulls torch."""
    # A subprocess, because this test session has already imported torch.
    script = "import sys, trackers; print('torch' in sys.modules)"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error::FutureWarning", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"
