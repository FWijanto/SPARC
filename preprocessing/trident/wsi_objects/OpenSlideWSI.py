from __future__ import annotations
import os
import re
import numpy as np
import openslide
from PIL import Image
from typing import List, Tuple, Union, Optional

from preprocessing.trident.wsi_objects.WSI import WSI, ReadMode


class _TifffileAdapter:
    """
    Minimal OpenSlide-compatible wrapper for BigTIFF SVS files (e.g. Leica GT450)
    that OpenSlide cannot open. Uses tifffile + zarr for tile reads.

    Shape convention for these files: (Z, Y, X, S) with axes ZYXS.
    We always read Z-plane 0.

    Fork-safe: detects when a DataLoader worker has forked and re-opens the
    TiffFile in the child process so each worker has its own file descriptor.
    """

    def __init__(self, path: str) -> None:
        import tifffile
        self._path = path

        # Read metadata once; keep no persistent file handle after __init__.
        with tifffile.TiffFile(path) as tif:
            levels = tif.series[0].levels

            def _wh(lev):
                return (lev.shape[-2], lev.shape[-3])  # (width=X, height=Y)

            level_dims = [_wh(lev) for lev in levels]
            self.dimensions = level_dims[0]
            self.level_count = len(levels)
            self.level_dimensions = tuple(level_dims)
            self.level_downsamples = tuple(
                self.dimensions[0] / w for w, _ in level_dims
            )

            desc = getattr(tif.pages[0], 'description', '') or ''
            self.properties = {}
            m = re.search(r'MPP\s*=\s*([\d.]+)', desc)
            if m:
                self.properties['openslide.mpp-x'] = m.group(1)
                self.properties['openslide.mpp-y'] = m.group(1)
                self.properties['aperio.MPP'] = m.group(1)
            m = re.search(r'AppMag\s*=\s*(\d+)', desc)
            if m:
                self.properties['openslide.objective-power'] = m.group(1)

        self._tif = None
        self._zarr_cache: list = [None] * self.level_count
        self._owner_pid: int = -1  # intentionally invalid; _ensure_open() runs on first use

    def _ensure_open(self) -> None:
        """Re-open TiffFile if we are in a new process (DataLoader fork)."""
        import tifffile
        pid = os.getpid()
        if pid != self._owner_pid:
            if self._tif is not None:
                self._tif.close()
            self._tif = tifffile.TiffFile(self._path)
            self._zarr_cache = [None] * self.level_count
            self._owner_pid = pid

    def _zarr(self, level: int):
        import zarr
        self._ensure_open()
        if self._zarr_cache[level] is None:
            store = self._tif.aszarr(series=0, level=level)
            self._zarr_cache[level] = zarr.open(store, mode='r')
        return self._zarr_cache[level]

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> Image.Image:
        x, y = location
        w, h = size
        ds = self.level_downsamples[level]
        lx = int(round(x / ds))
        ly = int(round(y / ds))
        lw, lh = self.level_dimensions[level]

        x1, y1 = min(lx + w, lw), min(ly + h, lh)
        arr = self._zarr(level)[0, ly:y1, lx:x1]  # (h', w', 3)

        img = Image.fromarray(arr.astype(np.uint8), 'RGB').convert('RGBA')
        result = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        result.paste(img, (0, 0))
        return result

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        arr = np.asarray(self._zarr(self.level_count - 1)[0])  # (H, W, 3)
        img = Image.fromarray(arr, 'RGB')
        img.thumbnail(size, Image.LANCZOS)
        return img.convert('RGB')

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
            self._tif = None


class OpenSlideWSI(WSI):

    def __init__(self, slide_path, **kwargs) -> None:
        """
        Initialize an OpenSlideWSI instance.

        Parameters
        ----------
        slide_path : str
            Path to the WSI file.
        **kwargs : dict
            Keyword arguments forwarded to the base `WSI` class. Most important key is:
            - lazy_init (bool, default=True): Whether to defer loading WSI and metadata.

        Please refer to WSI constructor for all parameters.

        Example
        -------
        >>> wsi = OpenSlideWSI(slide_path="path/to/wsi.svs", lazy_init=False)
        >>> print(wsi)
        <width=100000, height=80000, backend=OpenSlideWSI, mpp=0.25, mag=40>
        """
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using OpenSlide.

        This method opens a whole-slide image using the OpenSlide backend, extracting
        key metadata including dimensions, magnification, and multiresolution pyramid
        information. If a tissue segmentation mask is provided, it is also loaded.

        Raises
        ------
        FileNotFoundError
            If the WSI file or the tissue segmentation mask cannot be found.
        Exception
            If an unexpected error occurs during WSI initialization.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the base level.
        - `dimensions`: (width, height) tuple from the highest resolution.
        - `level_count`: number of resolution levels in the image pyramid.
        - `level_downsamples`: downsampling factors for each level.
        - `level_dimensions`: image dimensions at each level.
        - `properties`: metadata dictionary from OpenSlide.
        - `mpp`: microns per pixel, inferred if not manually specified.
        - `mag`: estimated magnification level.
        - `gdf_contours`: loaded from `tissue_seg_path` if provided.
        """

        super()._lazy_initialize()

        if not self.lazy_init:
            try:
                try:
                    self.img = openslide.OpenSlide(self.slide_path)
                except openslide.OpenSlideUnsupportedFormatError:
                    # BigTIFF SVS files (e.g. Leica GT450) are not supported by
                    # OpenSlide. Fall back to a tifffile-backed adapter.
                    self.img = _TifffileAdapter(self.slide_path)
                # set openslide attrs as self
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
                raise RuntimeError(f"Failed to initialize WSI with OpenSlide: {e}") from e

    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """
        Retrieve microns per pixel (MPP) from OpenSlide metadata.

        Parameters
        ----------
        custom_mpp_keys : list of str, optional
            Additional metadata keys to check for MPP.

        Returns
        -------
        float
            MPP value in microns per pixel.

        Raises
        ------
        ValueError
            If MPP cannot be determined from metadata.
        """
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            'openslide.mirax.MPP',
            'aperio.MPP',
            'hamamatsu.XResolution',
            'openslide.comment',
        ]

        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in self.img.properties:
                try:
                    mpp_x = float(self.img.properties[key])
                    return round(mpp_x, 4)
                except ValueError:
                    continue

        x_resolution = self.img.properties.get('tiff.XResolution')
        unit = self.img.properties.get('tiff.ResolutionUnit')

        if x_resolution and unit:
            try:
                if unit.lower() == 'centimeter':
                    return round(10000 / float(x_resolution), 4)
                elif unit.upper() == 'INCH':
                    return round(25400 / float(x_resolution), 4)
            except ValueError:
                pass

        raise ValueError(
            f"Unable to extract MPP from slide metadata: '{self.slide_path}'.\n"
            "Suggestions:\n"
            "- Provide `custom_mpp_keys` to specify metadata keys to look for.\n"
            "- Set the MPP explicitly via the class constructor.\n"
            "- If using the `run_batch_of_slides.py` script, pass the MPP via the "
            "`--custom_list_of_wsis` argument in a CSV file. Refer to TRIDENT/README/Q&A."
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:

        """
        Retrieve estimated magnification from metadata.

        Parameters
        ----------
        custom_mpp_keys : list of str, optional
            Keys to aid in computing magnification from MPP.

        Returns
        -------
        int
            Estimated magnification.

        Raises
        ------
        ValueError
            If magnification cannot be determined.
        """
        mag = super()._fetch_magnification(custom_mpp_keys)
        if mag is not None:
            return mag

        metadata_mag = self.img.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if metadata_mag is not None:
            try:
                return int(metadata_mag)
            except ValueError:
                pass

        raise ValueError(f"Unable to determine magnification from metadata for: {self.slide_path}")

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the whole-slide image (WSI).

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            (width, height) of the region to extract.
        read_as : {'pil', 'numpy'}, optional
            Output format for the region:
            - 'pil': returns a PIL Image (default)
            - 'numpy': returns a NumPy array (H, W, 3)

        Returns
        -------
        Union[PIL.Image.Image, np.ndarray]
            Extracted image region in the specified format.

        Raises
        ------
        ValueError
            If `read_as` is not one of 'pil' or 'numpy'.

        Example
        -------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512), read_as='numpy')
        >>> print(region.shape)
        (512, 512, 3)
        """
        region = self.img.read_region(location, level, size).convert('RGB')

        if read_as == 'pil':
            return region
        elif read_as == 'numpy':
            return np.array(region)
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the WSI.

        Returns
        -------
        tuple of int
            (width, height) in pixels.
        """
        return self.img.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the WSI.

        Parameters
        ----------
        size : tuple of int
            Desired (width, height) of the thumbnail.

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail as a PIL Image.
        """
        return self.img.get_thumbnail(size).convert('RGB')
