from mirm.model.core import IRTModel, BinaryIRTModel, MarginalBinaryIRTModel
from mirm.model.baselines import (
    RaschModel,
    TwoPLModel,
    ThreePLModel,
    LPEModel,
    RHModel,
    QuadModel,
    SplineModel,
)
from mirm.model.mirm import BinaryMiRM, CellMiRM
from mirm.model.prm import PRM

__all__ = [
    "IRTModel",
    "BinaryIRTModel",
    "MarginalBinaryIRTModel",
    "RaschModel",
    "TwoPLModel",
    "ThreePLModel",
    "LPEModel",
    "RHModel",
    "QuadModel",
    "SplineModel",
    "BinaryMiRM",
    "CellMiRM",
    "PRM",
]

MODEL_NAMES = {cls.name: cls for cls in (
    RaschModel,
    TwoPLModel,
    ThreePLModel,
    LPEModel,
    RHModel,
    QuadModel,
    SplineModel,
    BinaryMiRM,
)}

def build_model(name: str, n_items: int, n_quad: int = 41, **params) -> IRTModel:
    cls = MODEL_NAMES[name]
    if name == "Spline":
        return cls(n_items, params.get("n_basis", 18), n_quad)
    if name == "MiRM":
        return cls(n_items, params["c"], params["d"], n_quad)
    return cls(n_items, n_quad)