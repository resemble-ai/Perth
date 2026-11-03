"""
Perth: Audio Watermarking and Detection Library.

This library provides tools and algorithms for embedding and detecting
watermarks in audio files using various techniques.
"""

from .watermarker import WatermarkerBase, WatermarkingException
from .dummy_watermarker import DummyWatermarker
from .perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

# Make core classes/functions available at the package level
__all__ = [
    "WatermarkerBase",
    "WatermarkingException",
    "DummyWatermarker",
    "PerthImplicitWatermarker",
]

# Version information
__version__ = "1.0.0"
__author__ = "Resemble AI Team"
