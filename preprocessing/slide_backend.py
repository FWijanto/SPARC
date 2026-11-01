"""Shared slide-reader backend helper.

Adds first-class support for ``tiffslide`` alongside ``openslide`` throughout
the SPARC preprocessing/analysis pipeline.

``tiffslide`` is a quasi drop-in replacement for ``openslide``: its
``TiffSlide`` object mirrors the ``openslide.OpenSlide`` API
(``.dimensions``, ``.level_count``, ``.level_dimensions``,
``.level_downsamples``, ``.properties``, ``.read_region``, ``.get_thumbnail``).
"""

from __future__ import annotations

import os
from typing import Literal, Optional

SlideBackend = Literal["openslide", "tiffslide"]

OPENVSLIDE_LIKE = ("openslide", "tiffslide")


def _normalize_backend(reader_type: Optional[str]) -> str:
    """Lower-case / default a reader-type string.

    trident also knows ``image`` and ``cucim``; those pass through unchanged
    so non-openslide backends keep working exactly as before.
    """
    if not reader_type:
        return "openslide"
    return str(reader_type).lower()


def open_slide(path, backend: str = "openslide"):
    """Open a whole-slide image with the chosen backend.

    Used by code that talks to the raw slide object directly (``mpp.py`` MPP
    probing, ``niche_atlas.py`` patch viewing) rather than through trident's
    WSI/Processor stack.
    """
    rt = _normalize_backend(backend)
    if rt == "tiffslide":
        import tiffslide

        return tiffslide.TiffSlide(str(path))
    if rt == "openslide":
        import openslide

        return openslide.OpenSlide(str(path))
    raise ValueError(f"open_slide() only supports openslide/tiffslide, got {backend!r}")


# MPP property keys, openslide-first (unchanged behaviour)
# then tiffslide's own keys, then legacy vendor keys. tiffslide always fills
# ``tiffslide.mpp-x/y`` for files it can read, and additionally mirrors the
# legacy vendor keys it found in the file (e.g. ``aperio.MPP``).
MPP_KEYS_X = ("openslide.mpp-x", "tiffslide.mpp-x", "aperio.MPP", "openslide.mirax.MPP")
MPP_KEYS_Y = ("openslide.mpp-y", "tiffslide.mpp-y")


def _first_mpp(properties: dict, keys) -> Optional[float]:
    for key in keys:
        value = properties.get(key)
        if value is None:
            continue
        try:
            mpp = float(value)
        except (TypeError, ValueError):
            continue
        if mpp > 0:
            return mpp
    return None


def read_mpp_from_props(properties: dict) -> Optional[float]:
    """Average x/y microns-per-pixel from a slide's ``properties`` mapping.

    Works for both openslide and tiffslide: openslide keys win when present, so
    existing openslide slides resolve exactly as before. Returns ``None`` when
    MPP is absent or invalid.
    """
    x = _first_mpp(properties, MPP_KEYS_X)
    if x is None:
        return None
    y = _first_mpp(properties, MPP_KEYS_Y) or x
    return (x + y) / 2.0


# --- trident integration -------------------------------------------------

_patched = False


def ensure_tiffslide_registered() -> None:
    """Teach trident's ``load_wsi`` to accept ``reader_type="tiffslide"``.

    Idempotent and import-safe. Must be
    called before constructing a ``Processor`` with ``reader_type="tiffslide"``.
    """
    global _patched
    if _patched:
        return
    try:
        import trident
        from trident.wsi_objects.WSIFactory import load_wsi as _orig_load_wsi
        from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
        from trident.wsi_objects.WSI import WSI
    except Exception:
        return

    # Note: ``trident.__init__`` re-exports the ``Processor``/``WSIFactory``
    # *classes*, shadowing those submodules as package attributes. So always
    # import the real modules here and patch them via ``sys.modules`` --
    # that is where the ``load_wsi`` global that call sites resolve lives.
    try:
        import trident.Processor 
    except Exception:
        pass

    class TiffSlideWSI(OpenSlideWSI):
        """trident WSI backed by ``tiffslide`` instead of ``openslide``.

        Inherits every trident capability (mpp/mag inference, read_region,
        extract_patch, level bookkeeping) unchanged. Only the slide-open call
        and the MPP key list differ -- exactly the openslide/tiffslide
        divergence.
        """

        def _open_tiffslide(self) -> None:
            # Open the slide and remember which process
            # opened it.  tifffile reads through one file handle using
            # seek+read; when trident's feature extraction forks its DataLoader
            # workers, every child would otherwise share the parent's handle
            # and its file offset.  Interleaved reads then corrupt tile bytes
            # openslide never hits this because the C library reads via
            # offset-independent pread().
            import tiffslide

            self.img = tiffslide.TiffSlide(self.slide_path)
            self._opened_pid = os.getpid()

        def _ensure_process_local(self) -> None:
            # If a forked worker inherited the parent's
            # TiffSlide handle, reopen the slide so each process reads through
            # its own handle.
            if getattr(self, "img", None) is None:
                return
            if getattr(self, "_opened_pid", None) != os.getpid():
                self._open_tiffslide()

        def read_region(self, location, level, size, read_as="pil"):
            # Choke point used by WSIPatcher in every
            # forked worker: first make the handle process-local, then convert
            # any decode failure into a plain picklable RuntimeError.
            self._ensure_process_local()
            try:
                return super().read_region(location, level, size, read_as=read_as)
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"tiffslide read_region failed for {self.slide_path!r} "
                    f"level={level} location={tuple(location)} size={tuple(size)}: "
                    f"{type(e).__name__}: {e}"
                ) from e

        def _lazy_initialize(self) -> None:
            # Same body as trident's
            # OpenSlideWSI._lazy_initialize, but open with tiffslide.TiffSlide
            # Note: the placeholder setup calls
            # the base ``WSI`` directly and NOT super(), because
            # ``OpenSlideWSI._lazy_initialize`` would open the slide with
            # openslide and defeat this subclass.
            WSI._lazy_initialize(self)
            if not self.lazy_init:
                try:
                    # [TIFFSLIDE CHANGE] open through the process-tracking helper
                    self._open_tiffslide()
                    self.dimensions = self.get_dimensions()
                    self.width, self.height = self.dimensions
                    self.level_count = self.img.level_count
                    self.level_downsamples = self.img.level_downsamples
                    self.level_dimensions = self.img.level_dimensions
                    self.properties = self.img.properties
                    if self.mpp is None:
                        self.mpp = self._fetch_mpp(self.custom_mpp_keys)
                    self.mag = self._fetch_magnification(self.custom_mpp_keys)
                    self.lazy_init = True
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to initialize WSI with tiffslide: {e}"
                    ) from e

        def _fetch_mpp(self, custom_mpp_keys=None):
            # tiffslide may expose MPP under its own keys;
            # try the inherited openslide keys first, then tiffslide-specific
            # keys, then a tifffile description fallback.
            # tiffslide can include keys whose *value* is None. trident's
            # base implementation does float(props[key]) and crashes with an
            # unguarded TypeError on None, so drop None-valued keys before
            # delegating and catch TypeError as well as ValueError here.
            props = self.img.properties
            keys = [
                k for k in (list(custom_mpp_keys or []) + ["tiffslide.mpp-x", "tiffslide.mpp-y"])
                if props.get(k) is not None
            ]
            try:
                return super()._fetch_mpp(keys)
            except (ValueError, TypeError):
                mpp = read_mpp_from_props(props)
                if mpp is not None:
                    return mpp
                raise

    def _load_wsi_with_tiffslide(
        slide_path, reader_type=None, **kwargs
    ):
        rt = reader_type.lower() if isinstance(reader_type, str) else reader_type
        # Intercept tiffslide before delegating the rest to
        # trident's original factory (openslide / image / cucim / auto).
        if rt == "tiffslide":
            return TiffSlideWSI(slide_path=slide_path, **kwargs)
        return _orig_load_wsi(slide_path, reader_type, **kwargs)

    import sys

    # Patch the three module globals that call sites resolve:
    # - ``trident`` itself (``trident.load_wsi(...)`` style calls),
    # - the factory module (``trident.wsi_objects.WSIFactory.py``), reached via
    #   sys.modules because the package attribute of that name is the class,
    # - ``trident.Processor``, which bound ``from trident import load_wsi`` at
    #   import time and must be replaced independently of the factory module.
    for mod in (
        trident,
        sys.modules.get("trident.wsi_objects.WSIFactory"),
        sys.modules.get("trident.Processor"),
    ):
        if mod is not None and getattr(mod, "load_wsi", None) is _orig_load_wsi:
            mod.load_wsi = _load_wsi_with_tiffslide

    _patched = True
