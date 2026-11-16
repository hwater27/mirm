from importlib import import_module
import pkgutil
from typing import Any

from mirm.data.datasets import ScoredIRTDataset, KeyedIRTDataset


def available_preprocessors() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.iter_modules(__path__)
        if not module.name.startswith("_")
    )


def preprocess(name: str, **options: Any) -> tuple[ScoredIRTDataset | KeyedIRTDataset, dict]:
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Invalid preprocessor name: {name}")

    available = available_preprocessors()
    if name not in available:
        raise ValueError(f"Invalid preprocessor name: {name!r}. Available: {available}")

    module = import_module(f"{__name__}.{name}")
    processor = getattr(module, "preprocess", None)
    if not callable(processor):
        raise AttributeError(f"Preprocessor {name!r} does not have preprocess()")

    result = processor(**options)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("preprocess() must return (dataset, metadata).")

    dataset, metadata = result
    if not isinstance(dataset, (ScoredIRTDataset, KeyedIRTDataset)):
        raise TypeError("Preprocessor must return an IRT dataset.")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    return dataset, metadata


def inspect(name: str, metadata: dict, **options) -> dict:
    inspector = getattr(import_module(f"{__name__}.{name}"), "inspect", None)
    if not callable(inspector):
        raise AttributeError(f"Preprocessor {name!r} does not have inspect()")
    result = inspector(metadata, **options)
    if not isinstance(result, dict):
        raise TypeError("inspect() must return a dictionary.")
    return result