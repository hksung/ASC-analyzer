# asc_analyzer/__init__.py

"""
asc_analyzer: Argument‐Structure‐Construction analyzer package.
"""

__version__ = "1.0.0"

# asc_analyzer/__init__.py
"""
ASC Analyzer package: core functions to extract and compute ASC-based indices.
"""

__version__ = "0.1.0"

# Public API
from .core import (
    fullExtractSent,
    fullExtractDoc,
    processText,
    ttr,
    MATTR,
    proportion,
    freqLookup,
    soaLookup,
    indexCalc,
    indexCalcFull,
    writeCsv,
)

# Expose module-level API
__all__ = [
    "__version__",
    "fullExtractSent",
    "fullExtractDoc",
    "processText",
    "ttr",
    "mattr",
    "proportion",
    "freqLookup",
    "soaLookup",
    "indexCalc",
    "indexCalcFull",
    "writeCsv",
]
