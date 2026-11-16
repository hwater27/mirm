from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset


class _BaseIRTDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.student_ids = data[:, 0].long()
        self.item_ids = data[:, 1].long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        return self.data[idx]

    @property
    def n_students(self) -> int:
        return int(self.student_ids.max().item()) + 1

    @property
    def n_items(self) -> int:
        return int(self.item_ids.max().item()) + 1


class ScoredIRTDataset(_BaseIRTDataset):
    def __init__(self, data: torch.Tensor):
        super().__init__(data)

        self.credits = data[:, 2].to(torch.float64)
        if not torch.all((self.credits >= 0.0) & (self.credits <= 1.0)):
            raise ValueError("Credits must be between 0 and 1.")


class KeyedIRTDataset(_BaseIRTDataset):
    def __init__(self, data: torch.Tensor, answers: Sequence[int]):
        super().__init__(data)

        self.response_ids = data[:, 2].long()
        self.answers = torch.as_tensor(answers, dtype=torch.long)
        if min(self.response_ids) < 1:
            raise ValueError("Response IDs must be positive integers.")

    @property
    def n_choices(self) -> int:
        return int(torch.maximum(self.response_ids.max(), self.answers.max()).item())

    def to_scored(self, score_key: Optional[torch.Tensor] = None) -> ScoredIRTDataset:
        if score_key is None:
            credits = (self.response_ids == self.answers[self.item_ids]).to(torch.float64)
        else:
            score_key = torch.as_tensor(score_key)
            credits = score_key[self.item_ids, self.response_ids - 1]

        scored_data = torch.stack([
            self.student_ids.to(credits.dtype),
            self.item_ids.to(credits.dtype),
            credits,
        ], dim=1)

        return ScoredIRTDataset(scored_data)