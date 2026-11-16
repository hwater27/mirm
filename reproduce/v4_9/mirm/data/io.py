from pathlib import Path

import torch

from mirm.data.datasets import ScoredIRTDataset, KeyedIRTDataset


def save_dataset(
    path: str | Path,
    dataset: ScoredIRTDataset | KeyedIRTDataset,
    metadata: dict,
) -> None:
    metadata = dict(metadata)
    metadata.pop("cache_path", None)
    metadata.pop("artifact_dir", None)

    payload = {
        "kind": "keyed" if isinstance(dataset, KeyedIRTDataset) else "scored",
        "data": dataset.data.cpu(),
        "metadata": metadata,
    }

    if isinstance(dataset, KeyedIRTDataset):
        payload["answers"] = dataset.answers.cpu()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_dataset(path: str | Path) -> tuple[ScoredIRTDataset | KeyedIRTDataset, dict]:
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError("Dataset file must contain a dictionary payload.")

    if "kind" in payload:
        kind = payload["kind"]
        data = payload["data"]
        metadata = dict(payload.get("metadata") or {})
        answers = payload.get("answers")
    elif "data_tensor" in payload:
        kind = "scored"
        data = payload["data_tensor"]
        metadata = dict(payload.get("meta") or payload.get("metadata") or {})
        answers = payload.get("answers")
    else:
        raise ValueError("Dataset file is missing dataset contents.")

    metadata.pop("cache_path", None)
    metadata["artifact_dir"] = str(path.parent)

    if kind == "scored":
        dataset = ScoredIRTDataset(data)
    elif kind == "keyed":
        dataset = KeyedIRTDataset(data, answers)
    else:
        raise ValueError(f"Invalid dataset kind: {kind!r}")

    return dataset, metadata
