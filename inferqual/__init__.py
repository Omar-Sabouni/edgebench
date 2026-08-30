"""InferQual: a compact inference benchmark."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("inference-qualification")
except PackageNotFoundError:  # Source trees remain importable before installation.
    __version__ = "0.4.0"

__all__ = ["__version__"]
