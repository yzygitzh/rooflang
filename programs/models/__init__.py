"""Model implementations available to rooflang experiments."""

from importlib import import_module


MODEL_NAMES = ("dsv4_pro",)


def load_model(name):
    """Load a model package selected by an experiment CLI."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    return import_module(f"rooflang.programs.models.{name}")
