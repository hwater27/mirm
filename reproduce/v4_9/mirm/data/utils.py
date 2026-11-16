from typing import Sequence

import torch

from mirm.data.datasets import ScoredIRTDataset


def _reindex_students(dataset: ScoredIRTDataset) -> tuple[ScoredIRTDataset, torch.Tensor]:
    old_ids, new_ids = torch.unique(dataset.student_ids, sorted=True, return_inverse=True)

    data = dataset.data.clone()
    data[:, 0] = new_ids.to(data.dtype)

    return ScoredIRTDataset(data), old_ids


def _reindex_items(dataset: ScoredIRTDataset) -> tuple[ScoredIRTDataset, torch.Tensor]:
    old_ids, new_ids = torch.unique(dataset.item_ids, sorted=True, return_inverse=True)

    data = dataset.data.clone()
    data[:, 1] = new_ids.to(data.dtype)

    return ScoredIRTDataset(data), old_ids


def select_students(
    dataset: ScoredIRTDataset,
    student_ids: Sequence[int] | torch.Tensor,
    reindex: bool = True,
) -> tuple[ScoredIRTDataset, torch.Tensor]:
    selected_ids = torch.as_tensor(student_ids, dtype=torch.long, device=dataset.student_ids.device).unique(sorted=True)

    available = torch.unique(dataset.student_ids)
    if not torch.isin(selected_ids, available).all():
        raise ValueError("Some student IDs are not in the dataset.")

    mask = torch.isin(dataset.student_ids, selected_ids)
    selected = ScoredIRTDataset(dataset.data[mask].clone())

    if reindex:
        return _reindex_students(selected)

    return selected, selected_ids


def select_items(
    dataset: ScoredIRTDataset,
    item_ids: Sequence[int] | torch.Tensor,
    reindex: bool = True,
) -> tuple[ScoredIRTDataset, torch.Tensor]:
    selected_ids = torch.as_tensor(item_ids, dtype=torch.long, device=dataset.item_ids.device).unique(sorted=True)

    available = torch.unique(dataset.item_ids)
    if not torch.isin(selected_ids, available).all():
        raise ValueError("Some item IDs are not in the dataset.")

    mask = torch.isin(dataset.item_ids, selected_ids)
    selected = ScoredIRTDataset(dataset.data[mask].clone())

    if reindex:
        return _reindex_items(selected)

    return selected, selected_ids


def split_students(dataset: ScoredIRTDataset, test_ratio: float = 0.2, seed: int = 27) -> dict:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between zero and one.")

    student_ids = torch.unique(dataset.student_ids, sorted=True)
    n_students = len(student_ids)

    if n_students < 2:
        raise ValueError("At least two students are required.")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_students, generator=generator, device="cpu").to(student_ids.device)

    n_test = max(1, min(n_students - 1, round(test_ratio * n_students)))

    test_ids = student_ids[order[:n_test]]
    train_ids = student_ids[order[n_test:]]

    train, train_map = select_students(dataset, train_ids, reindex=True)
    test, test_map = select_students(dataset, test_ids, reindex=True)

    return {
        "train": train,
        "test": test,
        "train_student_ids": train_map,
        "test_student_ids": test_map,
    }

def top_items(dataset, n_items: int) -> tuple:
    """Keep the n_items items with the most responses (ties -> lower original index) and reindex items and students.

    Returns (subset, old_item_ids, old_student_ids): subset item i is original item old_item_ids[i] (ascending
    original order, so the selection is deterministic). Works for scored and keyed datasets alike, because the row
    filter and the reindexing depend only on the student and item columns.
    """
    counts = torch.bincount(dataset.item_ids, minlength=dataset.n_items)
    if n_items >= int((counts > 0).sum()):
        return dataset, torch.nonzero(counts > 0).flatten(), torch.unique(dataset.student_ids, sorted=True)

    order = sorted(range(len(counts)), key=lambda j: (-int(counts[j]), j))
    old_items = torch.tensor(sorted(order[:n_items]), dtype=torch.long)

    mask = torch.isin(dataset.item_ids, old_items)
    data = dataset.data[mask].clone()
    old_students, new_students = torch.unique(data[:, 0].long(), sorted=True, return_inverse=True)
    _, new_items = torch.unique(data[:, 1].long(), sorted=True, return_inverse=True)
    data[:, 0] = new_students.to(data.dtype)
    data[:, 1] = new_items.to(data.dtype)

    if hasattr(dataset, "answers"):
        subset = type(dataset)(data, dataset.answers[old_items])
    else:
        subset = type(dataset)(data)
    return subset, old_items, old_students
