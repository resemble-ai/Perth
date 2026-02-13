"""
Perth: Audio Watermarking and Detection Library.

This library provides tools and algorithms for embedding and detecting
watermarks in audio files using various techniques.
"""

from .watermarker import WatermarkerBase, WatermarkingException
from .dummy_watermarker import DummyWatermarker

# Import specific watermarker implementations
try:
    from .perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
except ImportError:
    PerthImplicitWatermarker = None

# Make core classes/functions available at the package level
__all__ = [
    "WatermarkerBase",
    "WatermarkingException",
    "DummyWatermarker",
]

# Add watermarker implementations if available
if PerthImplicitWatermarker is not None:
    __all__.append("PerthImplicitWatermarker")

# Version information
__version__ = "1.0.2"
__author__ = "Resemble AI Team"
