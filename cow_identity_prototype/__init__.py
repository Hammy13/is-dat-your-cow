"""Proof-of-concept hybrid cow identity package."""

from .config import PrototypeConfig, load_config
from .pipeline import CowIdentityPipeline

__all__ = ["PrototypeConfig", "load_config", "CowIdentityPipeline"]
