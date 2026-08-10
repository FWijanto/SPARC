from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trident")
except PackageNotFoundError:
    __version__ = "unknown"

from preprocessing.trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from preprocessing.trident.wsi_objects.CuCIMWSI import CuCIMWSI
from preprocessing.trident.wsi_objects.ImageWSI import ImageWSI
from preprocessing.trident.wsi_objects.WSIFactory import load_wsi, WSIReaderType
from preprocessing.trident.wsi_objects.WSIPatcher import OpenSlideWSIPatcher, WSIPatcher
from preprocessing.trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset

from preprocessing.trident.Visualization import visualize_heatmap

from preprocessing.trident.Processor import Processor

from preprocessing.trident.Converter import AnyToTiffConverter

from preprocessing.trident.Maintenance import deprecated

__all__ = [
    "Processor",
    "load_wsi",
    "OpenSlideWSI",
    "ImageWSI",
    "CuCIMWSI",
    "WSIPatcher",
    "OpenSlideWSIPatcher",
    "WSIPatcherDataset",
    "visualize_heatmap",
    "AnyToTiffConverter",
    "deprecated",
    "WSIReaderType",
]
