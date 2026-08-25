"""Experiments package - Experiment runners."""

from .precompute_spi import main as precompute_spi
from .train_autoencoder import main as train_autoencoder
from .grid_search import GridSearch

# Variable importance analysis - optional
try:
    from .analyze_variable_importance import analyze_variable_importance
except (ImportError, AttributeError):
    def analyze_variable_importance(*args, **kwargs):
        print("⚠️ Variable importance analysis not available")
        return None

__all__ = [
    "precompute_spi",
    "train_autoencoder",
    "GridSearch",
    "analyze_variable_importance",
]