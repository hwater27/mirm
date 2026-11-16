from importlib.metadata import PackageNotFoundError, version

import torch

# every real-valued tensor is float64 (training resolution).
torch.set_default_dtype(torch.float64)
# sparse invariants are checked where the response design is built (model.core.response_design);
# every other sparse operation runs on those checked designs, so the global check is off.
torch.sparse.check_sparse_tensor_invariants.disable()


try:
    __version__ = version("mirm")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = ["__version__"]
