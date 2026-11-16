import csv
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from mirm.data.datasets import KeyedIRTDataset


CHOICE_MAP = {"a": 1, "b": 2, "c": 3, "d": 4}
PREPROCESSOR_VERSION = 1
SCHEMA_VERSION = "ednet-kt1-keyed-v1"


def _numbered_id(value: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", value)
    if match is None:
        raise ValueError(f"Invalid EdNet identifier: {value!r}")
    return int(match.group(1))


def _cache_path(output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    if path.suffix.casefold() != ".pt":
        path = path / "ednet_kt1.pt"
    return path


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _load_questions(
    questions_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(questions_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    questions = pd.read_csv(
        path,
        dtype={"question_id": "string", "correct_answer": "string"},
    )
    required = {"question_id", "correct_answer"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(
            f"questions.csv is missing columns: {sorted(missing)}"
        )
    if questions["question_id"].isna().any():
        raise ValueError("questions.csv contains missing question IDs.")

    questions["question_id"] = questions["question_id"].str.strip()
    if questions["question_id"].duplicated().any():
        raise ValueError("questions.csv contains duplicate question IDs.")

    questions["correct_answer"] = (
        questions["correct_answer"].str.strip().str.lower()
    )
    invalid = ~questions["correct_answer"].isin(CHOICE_MAP)
    if invalid.any():
        values = sorted(
            questions.loc[invalid, "correct_answer"]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"questions.csv contains invalid correct answers: {values}"
        )

    for question_id in questions["question_id"]:
        _numbered_id(str(question_id), "q")

    answer_key = {
        str(question_id): CHOICE_MAP[str(answer)]
        for question_id, answer in zip(
            questions["question_id"],
            questions["correct_answer"],
        )
    }
    return questions, answer_key


def _student_files(
    kt1_path: str | Path,
    n_students: Optional[int],
    seed: int,
) -> list[Path]:
    root = Path(kt1_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    csv_files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.casefold() == ".csv"
    ]
    malformed = [
        path.name
        for path in csv_files
        if re.fullmatch(r"u\d+", path.stem) is None
    ]
    if malformed:
        raise ValueError(
            f"KT1 contains malformed student filenames: {malformed[:10]}"
        )

    files = csv_files
    files.sort(key=lambda path: _numbered_id(path.stem, "u"))

    if not files:
        raise ValueError(f"No KT1 student files found in {root}.")
    if n_students is not None:
        if n_students < 1:
            raise ValueError("n_students must be positive.")
        if n_students < len(files):
            generator = np.random.default_rng(seed)
            selected = np.sort(
                generator.choice(
                    len(files),
                    size=n_students,
                    replace=False,
                )
            )
            files = [files[index] for index in selected]

    return files


def _load_interactions(
    files: Sequence[Path],
    answer_key: dict[str, int],
    first_response_only: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        "timestamp",
        "solving_id",
        "question_id",
        "user_answer",
        "elapsed_time",
    }
    frames: list[pd.DataFrame] = []
    n_unknown_questions = 0
    n_invalid_answers = 0
    n_repeated_responses = 0

    for path in files:
        frame = pd.read_csv(
            path,
            dtype={
                "question_id": "string",
                "user_answer": "string",
            },
        )
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        user_id = path.stem
        _numbered_id(user_id, "u")
        frame = frame.loc[:, list(required)].copy()
        for column in ("timestamp", "solving_id", "elapsed_time"):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or not np.equal(
                numeric,
                np.floor(numeric),
            ).all():
                raise ValueError(
                    f"{path.name} contains invalid {column} values."
                )
            frame[column] = numeric.astype(np.int64)

        frame["user_id"] = user_id
        frame["source_row"] = np.arange(len(frame), dtype=np.int64)
        frame["question_id"] = frame["question_id"].str.strip()
        frame["user_answer"] = (
            frame["user_answer"].str.strip().str.lower()
        )

        known = frame["question_id"].isin(answer_key)
        n_unknown_questions += int((~known).sum())
        frame = frame.loc[known].copy()

        repeated = frame.duplicated(
            subset=["question_id"],
            keep="first",
        )
        n_repeated_responses += int(repeated.sum())
        if first_response_only:
            frame = (
                frame.sort_values(
                    ["timestamp", "source_row"],
                    kind="stable",
                )
                .drop_duplicates(
                    subset=["question_id"],
                    keep="first",
                )
            )

        valid_answer = frame["user_answer"].isin(CHOICE_MAP)
        n_invalid_answers += int((~valid_answer).sum())
        frame = frame.loc[valid_answer].copy()
        frame["response_id"] = (
            frame["user_answer"].map(CHOICE_MAP).astype(np.int64)
        )
        frames.append(frame)

    if not frames:
        raise ValueError("No KT1 interactions were loaded.")

    interactions = pd.concat(frames, ignore_index=True)
    if interactions.empty:
        raise ValueError("No valid KT1 interactions remain.")

    return interactions, {
        "student_files": len(files),
        "unknown_question_rows": n_unknown_questions,
        "invalid_answer_rows": n_invalid_answers,
        "repeated_response_rows": n_repeated_responses,
    }


def _stable_filter(
    interactions: pd.DataFrame,
    min_items_per_student: int,
    min_students_per_item: int,
) -> pd.DataFrame:
    if min_items_per_student < 0 or min_students_per_item < 0:
        raise ValueError("Minimum response counts cannot be negative.")

    required_student = max(1, int(min_items_per_student))
    required_item = max(1, int(min_students_per_item))
    filtered = interactions

    while not filtered.empty:
        pairs = filtered[
            ["user_id", "question_id"]
        ].drop_duplicates()
        student_counts = pairs["user_id"].value_counts()
        item_counts = pairs["question_id"].value_counts()
        keep_students = student_counts[
            student_counts >= required_student
        ].index
        keep_items = item_counts[
            item_counts >= required_item
        ].index
        updated = filtered[
            filtered["user_id"].isin(keep_students)
            & filtered["question_id"].isin(keep_items)
        ]

        if len(updated) == len(filtered):
            return filtered
        filtered = updated

    raise ValueError("No interactions remain after iterative filtering.")


def _top_ids(
    interactions: pd.DataFrame,
    id_column: str,
    other_column: str,
    count: int,
) -> list[str]:
    if count < 1:
        raise ValueError("top-k values must be positive.")

    counts = (
        interactions[[id_column, other_column]]
        .drop_duplicates()[id_column]
        .value_counts()
        .rename("count")
        .rename_axis(id_column)
        .reset_index()
    )
    prefix = "u" if id_column == "user_id" else "q"
    counts["numeric_id"] = counts[id_column].map(
        lambda value: _numbered_id(str(value), prefix)
    )
    counts = counts.sort_values(
        ["count", "numeric_id"],
        ascending=[False, True],
        kind="stable",
    )
    return counts[id_column].head(count).astype(str).tolist()


def _save_artifacts(
    output_path: Path,
    interactions: pd.DataFrame,
    questions: pd.DataFrame,
    students: Sequence[str],
    items: Sequence[str],
    item_to_idx: dict[str, int],
    answer_key: dict[str, int],
) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    student_path = output_path.with_suffix(".students.npz")
    np.savez_compressed(
        student_path,
        student_idx=np.arange(len(students), dtype=np.int64),
        user_id=np.asarray(students, dtype=str),
    )

    item_path = output_path.with_suffix(".items.csv")
    selected_questions = questions[
        questions["question_id"].isin(items)
    ].copy()
    selected_questions["item_idx"] = selected_questions[
        "question_id"
    ].map(item_to_idx)
    selected_questions["answer_id"] = selected_questions[
        "question_id"
    ].map(answer_key)
    selected_questions = selected_questions.sort_values("item_idx")
    selected_questions.to_csv(item_path, index=False)

    response_path = output_path.with_suffix(".responses.npz")
    np.savez_compressed(
        response_path,
        timestamp=interactions["timestamp"].to_numpy(),
        solving_id=interactions["solving_id"].to_numpy(),
        elapsed_time=interactions["elapsed_time"].to_numpy(),
        source_row=interactions["source_row"].to_numpy(),
    )

    return {
        "students": student_path.name,
        "items": item_path.name,
        "responses": response_path.name,
    }


def _preprocess_ednet(
    kt1_path: str | Path,
    questions_path: str | Path,
    output_path: str | Path,
    *,
    n_students: Optional[int] = None,
    min_students_per_item: int = 200,
    min_items_per_student: int = 20,
    top_k_students: Optional[int] = None,
    top_k_items: Optional[int] = None,
    first_response_only: bool = True,
    seed: int = 27,
    save_meta: bool = True,
) -> tuple[KeyedIRTDataset, dict]:
    questions, answer_key = _load_questions(questions_path)
    files = _student_files(kt1_path, n_students, seed)
    interactions, audit = _load_interactions(
        files,
        answer_key,
        first_response_only,
    )
    interactions = _stable_filter(
        interactions,
        min_items_per_student,
        min_students_per_item,
    )

    if top_k_items is not None:
        keep_items = _top_ids(
            interactions,
            "question_id",
            "user_id",
            top_k_items,
        )
        interactions = interactions[
            interactions["question_id"].isin(keep_items)
        ]
    if top_k_students is not None:
        keep_students = _top_ids(
            interactions,
            "user_id",
            "question_id",
            top_k_students,
        )
        interactions = interactions[
            interactions["user_id"].isin(keep_students)
        ]
    if top_k_items is not None or top_k_students is not None:
        interactions = _stable_filter(
            interactions,
            min_items_per_student,
            min_students_per_item,
        )

    students = sorted(
        interactions["user_id"].unique().tolist(),
        key=lambda value: _numbered_id(str(value), "u"),
    )
    items = sorted(
        interactions["question_id"].unique().tolist(),
        key=lambda value: _numbered_id(str(value), "q"),
    )
    student_to_idx = {
        str(user_id): index
        for index, user_id in enumerate(students)
    }
    item_to_idx = {
        str(question_id): index
        for index, question_id in enumerate(items)
    }

    interactions = interactions.copy()
    interactions["student_idx"] = interactions["user_id"].map(
        student_to_idx
    )
    interactions["item_idx"] = interactions["question_id"].map(
        item_to_idx
    )
    data = torch.from_numpy(
        interactions[
            ["student_idx", "item_idx", "response_id"]
        ].to_numpy(dtype=np.int64)
    )
    answers = torch.tensor(
        [answer_key[str(question_id)] for question_id in items],
        dtype=torch.long,
    )
    dataset = KeyedIRTDataset(data, answers)

    correct = (
        dataset.response_ids
        == dataset.answers[dataset.item_ids]
    )
    output = _cache_path(output_path)
    metadata: dict[str, Any] = {
        "preprocessor": "ednet",
        "preprocessor_version": PREPROCESSOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_students": len(students),
        "n_items": len(items),
        "n_choices": dataset.n_choices,
        "n_responses": len(dataset),
        "accuracy": float(correct.float().mean().item()),
        "selection": {
            "n_students_requested": n_students,
            "min_students_per_item": min_students_per_item,
            "min_items_per_student": min_items_per_student,
            "top_k_students": top_k_students,
            "top_k_items": top_k_items,
            "first_response_only": first_response_only,
            "seed": seed,
        },
        "encoding": {
            "response_ids": {
                choice: response_id
                for choice, response_id in CHOICE_MAP.items()
            },
            "response_id_origin": 1,
        },
        "audit": audit,
    }

    if save_meta:
        metadata["attribute_files"] = _save_artifacts(
            output,
            interactions,
            questions,
            students,
            items,
            item_to_idx,
            answer_key,
        )
        output.with_suffix(".meta.json").write_text(
            json.dumps(
                _jsonable(metadata),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "keyed",
            "data": dataset.data.cpu(),
            "answers": dataset.answers.cpu(),
            "metadata": metadata,
        },
        output,
    )
    return dataset, metadata


def preprocess(**options: Any) -> tuple[KeyedIRTDataset, dict]:
    required = {"kt1_path", "questions_path", "output_path"}
    missing = required - set(options)
    if missing:
        raise TypeError(
            f"EdNet preprocessing requires: {sorted(missing)}"
        )
    return _preprocess_ednet(**options)


def _selected_ids(
    values: Optional[Sequence[int]],
    size: int,
    label: str,
) -> list[int]:
    if values is None:
        return []

    selected = [int(value) for value in values]
    if any(value < 0 or value >= size for value in selected):
        raise IndexError(
            f"{label} IDs must be between 0 and {size - 1}."
        )
    return selected


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _artifact_path(
    metadata: dict,
    key: str,
    artifact_dir: Optional[str | Path],
) -> Path:
    files = metadata.get("attribute_files")
    if not isinstance(files, dict) or key not in files:
        raise ValueError(
            "Inspection metadata is unavailable. "
            "Run preprocessing with save_meta=True."
        )

    base = artifact_dir
    if base is None:
        base = metadata.get("artifact_dir")
    if base is None:
        raise ValueError(
            "artifact_dir is required to locate inspection files."
        )

    path = Path(base) / files[key]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def inspect(
    metadata: dict,
    *,
    student_ids: Optional[Sequence[int]] = None,
    item_ids: Optional[Sequence[int]] = None,
    response_ids: Optional[Sequence[int]] = None,
    artifact_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    students: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    if student_ids is not None:
        path = _artifact_path(metadata, "students", artifact_dir)
        with np.load(path, allow_pickle=False) as attributes:
            selected = _selected_ids(
                student_ids,
                len(attributes["student_idx"]),
                "Student",
            )
            for student_id in selected:
                students.append({
                    name: _python_value(values[student_id])
                    for name, values in attributes.items()
                })

    if item_ids is not None:
        path = _artifact_path(metadata, "items", artifact_dir)
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        selected = _selected_ids(item_ids, len(rows), "Item")
        items = [dict(rows[item_id]) for item_id in selected]

    if response_ids is not None:
        path = _artifact_path(metadata, "responses", artifact_dir)
        with np.load(path, allow_pickle=False) as attributes:
            if not attributes.files:
                raise ValueError("Response attributes are empty.")
            n_responses = len(attributes[attributes.files[0]])
            if any(
                len(attributes[name]) != n_responses
                for name in attributes.files
            ):
                raise ValueError(
                    "Response attribute lengths are inconsistent."
                )

            selected = _selected_ids(
                response_ids,
                n_responses,
                "Response",
            )
            for response_id in selected:
                record = {"response_idx": response_id}
                record.update({
                    name: _python_value(values[response_id])
                    for name, values in attributes.items()
                })
                responses.append(record)

    return {
        "students": students,
        "items": items,
        "responses": responses,
    }
