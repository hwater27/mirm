import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from mirm.data.datasets import ScoredIRTDataset


pyreadstat = import_module("pyreadstat")


Domain = Literal["R", "M", "S", "full"]
AdministrationScope = Literal["cba", "pba", "union", "intersection"]
PREPROCESSOR_VERSION = 4
SCHEMA_VERSION = "pisa2015-binary-v1"
SPECIAL_MISSING = {"M", "R", "N", "I", "V"}


@dataclass(frozen=True)
class SourceItem:
    source_name: str
    source_column_id: int
    item_local_id: int
    item_key: str
    label: str
    delivery_mode: str
    domain: str
    format_name: str
    no_credit_codes: frozenset[float]
    partial_credit_codes: frozenset[float]
    full_credit_codes: frozenset[float]

    @property
    def valid_codes(self) -> frozenset[float]:
        return (
            self.no_credit_codes
            | self.partial_credit_codes
            | self.full_credit_codes
        )


@dataclass(frozen=True)
class IndexedResponses:
    data_tensor: torch.Tensor
    student_ids: np.ndarray
    item_local_ids: np.ndarray
    raw_scores: np.ndarray
    source_column_ids: np.ndarray


def _iterative_response_filter(
    student_ids: np.ndarray,
    item_ids: np.ndarray,
    *,
    min_items_per_student: int = 0,
    min_students_per_item: int = 0,
) -> np.ndarray:
    if student_ids.ndim != 1 or item_ids.ndim != 1:
        raise ValueError("student_ids and item_ids must be one-dimensional")
    if len(student_ids) != len(item_ids):
        raise ValueError("student_ids and item_ids must have equal length")
    if min_items_per_student < 0 or min_students_per_item < 0:
        raise ValueError("minimum counts cannot be negative")

    active = np.ones(len(student_ids), dtype=bool)
    required_student = max(1, int(min_items_per_student))
    required_item = max(1, int(min_students_per_item))

    while active.any():
        active_students = student_ids[active]
        active_items = item_ids[active]
        unique_students, student_counts = np.unique(
            active_students,
            return_counts=True,
        )
        keep_students = unique_students[student_counts >= required_student]
        max_item = int(item_ids.max(initial=-1))
        item_counts = np.bincount(active_items, minlength=max_item + 1)
        keep_items = item_counts >= required_item
        updated = np.isin(student_ids, keep_students)
        updated &= keep_items[item_ids]

        if np.array_equal(updated, active):
            return active
        active = updated

    return active


def _index_responses(
    student_ids: np.ndarray,
    item_local_ids: np.ndarray,
    responses: np.ndarray,
    raw_scores: np.ndarray,
    source_column_ids: np.ndarray,
    active: np.ndarray,
) -> IndexedResponses:
    if not (
        len(student_ids)
        == len(item_local_ids)
        == len(responses)
        == len(raw_scores)
        == len(source_column_ids)
        == len(active)
    ):
        raise ValueError("all response arrays must have equal length")
    if not active.any():
        raise ValueError("no responses remain after filtering")

    selected_students = student_ids[active]
    selected_items = item_local_ids[active]
    unique_students = np.unique(selected_students)
    unique_items = np.unique(selected_items)
    student_idx = np.searchsorted(unique_students, selected_students)

    item_lookup = np.full(
        int(item_local_ids.max(initial=-1)) + 1,
        -1,
        dtype=np.int64,
    )
    item_lookup[unique_items] = np.arange(len(unique_items), dtype=np.int64)
    item_idx = item_lookup[selected_items]
    binary = responses[active].astype(np.int64, copy=False)
    data = np.column_stack((student_idx, item_idx, binary))

    return IndexedResponses(
        data_tensor=torch.from_numpy(data.astype(np.int64, copy=False)),
        student_ids=unique_students,
        item_local_ids=unique_items,
        raw_scores=raw_scores[active],
        source_column_ids=source_column_ids[active],
    )


def _validate_binary_dataset(
    data_tensor: torch.Tensor,
    answers: Sequence[int],
) -> None:
    if data_tensor.dtype != torch.long:
        raise ValueError(f"data_tensor must be torch.long, got {data_tensor.dtype}")
    if data_tensor.ndim != 2 or data_tensor.shape[1] != 3:
        raise ValueError("data_tensor must have shape [N, 3]")
    if data_tensor.shape[0] == 0:
        raise ValueError("data_tensor cannot be empty")
    if int(data_tensor.min()) < 0:
        raise ValueError("data_tensor cannot contain negative values")

    responses = data_tensor[:, 2]
    if not bool(torch.all((responses == 0) | (responses == 1))):
        raise ValueError("binary responses must be 0 or 1")

    students = torch.unique(data_tensor[:, 0], sorted=True)
    items = torch.unique(data_tensor[:, 1], sorted=True)
    if not torch.equal(students, torch.arange(len(students))):
        raise ValueError("student indices must be contiguous from zero")
    if not torch.equal(items, torch.arange(len(items))):
        raise ValueError("item indices must be contiguous from zero")
    if len(answers) != len(items) or any(int(value) != 1 for value in answers):
        raise ValueError("answers must contain one entry of 1 per item")

    encoded = (
        data_tensor[:, 0].numpy() * len(items)
        + data_tensor[:, 1].numpy()
    )
    encoded.sort()
    if len(encoded) > 1 and bool(np.any(encoded[1:] == encoded[:-1])):
        raise ValueError("duplicate (student_idx, item_idx) responses detected")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _save_compatible_cache(
    path: Path,
    data_tensor: torch.Tensor,
    answers: list[int],
    metadata: dict[str, Any],
    *,
    detailed_metadata: Optional[dict[str, Any]] = None,
    save_meta: bool = True,
) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_metadata = dict(metadata)
    if save_meta and detailed_metadata:
        cache_metadata.update(detailed_metadata)

    torch.save(
        {
            "data_tensor": data_tensor,
            "answers": answers,
            "meta": cache_metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )
    if save_meta:
        path.with_suffix(".meta.json").write_text(
            json.dumps(_jsonable(cache_metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return path


def _find_named(root: Path, filename: str) -> Path:
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.casefold() == filename.casefold()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _resolve_source(input_data_path: str | Path) -> tuple[Path, Path, Path]:
    value = Path(input_data_path).resolve()
    if value.is_file():
        if value.name.casefold() != "cy6_ms_cmb_stu_cog.sas7bdat":
            raise ValueError("A file input must be CY6_MS_CMB_STU_COG.sas7bdat")
        cog_path = value
        source_root = value.parents[1]
    elif value.is_dir():
        source_root = value
        cog_path = _find_named(source_root, "cy6_ms_cmb_stu_cog.sas7bdat")
    else:
        raise FileNotFoundError(value)

    format_path = _find_named(
        source_root,
        "CY6_MS_CMB_STU_COG.sas7bdat.format.sas",
    )
    return source_root, cog_path, format_path


def _parse_format_script(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    text = path.read_text(encoding="latin-1")
    formats: dict[str, dict[str, str]] = {}
    for match in re.finditer(
        r"(?ims)^\s*value\s+(\$?[A-Za-z_][A-Za-z0-9_]*)\s*(.*?)^\s*;",
        text,
    ):
        name, body = match.groups()
        entries: dict[str, str] = {}
        for entry in re.finditer(
            r'(?m)^\s*(.+?)\s*=\s*"([^"]*)"\s*$',
            body,
        ):
            code, label = entry.groups()
            entries[code.strip()] = label.strip()
        formats[name] = entries

    assignments: dict[str, str] = {}
    format_block = re.search(r"(?ims)^\s*format\s+(.*?)^\s*;", text)
    if not format_block:
        raise ValueError(f"No variable format assignment block found in {path}")
    for line in format_block.group(1).splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            assignments[parts[0]] = parts[1].rstrip(".")

    return formats, assignments


def _numeric_code(code: str) -> Optional[float]:
    raw = code.strip().strip('"').strip("'")
    if not raw or raw.startswith("."):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _score_codes(
    entries: dict[str, str],
) -> tuple[set[float], set[float], set[float]]:
    no_credit: set[float] = set()
    partial: set[float] = set()
    full: set[float] = set()

    for raw_code, label in entries.items():
        code = _numeric_code(raw_code)
        if code is None:
            continue
        normalized = label.casefold()
        if "full credit" in normalized:
            full.add(code)
        elif "partial credit" in normalized:
            partial.add(code)
        elif "no credit" in normalized:
            no_credit.add(code)

    return no_credit, partial, full


def _build_item_catalog(
    cog_path: Path,
    format_path: Path,
    domain: Domain,
    administration_scope: AdministrationScope,
) -> tuple[list[SourceItem], list[dict[str, Any]]]:
    _, metadata = pyreadstat.read_sas7bdat(
        cog_path,
        metadataonly=True,
        user_missing=True,
    )
    labels = dict(zip(metadata.column_names, metadata.column_labels))
    scored = [
        name
        for name, label in labels.items()
        if label and "Scored Response" in label and name[:1] in {"C", "P"}
    ]
    if domain != "full":
        scored = [
            name
            for name in scored
            if len(name) > 1 and name[1] == domain
        ]

    c_stems = {name[1:-1] for name in scored if name.startswith("C")}
    p_stems = {name[1:-1] for name in scored if name.startswith("P")}
    if administration_scope == "cba":
        selected = [name for name in scored if name.startswith("C")]
    elif administration_scope == "pba":
        selected = [name for name in scored if name.startswith("P")]
    elif administration_scope == "intersection":
        shared = c_stems & p_stems
        selected = [name for name in scored if name[1:-1] in shared]
    elif administration_scope == "union":
        selected = scored
    else:
        raise ValueError(
            f"Unsupported administration_scope: {administration_scope}"
        )

    formats, assignments = _parse_format_script(format_path)
    item_key_to_local: dict[str, int] = {}
    item_rows: dict[int, dict[str, Any]] = {}
    source_items: list[SourceItem] = []

    for source_column_id, name in enumerate(selected):
        item_key = (
            name[:-1]
            if administration_scope in {"cba", "pba"}
            else name[1:-1]
        )
        if item_key not in item_key_to_local:
            local_id = len(item_key_to_local)
            item_key_to_local[item_key] = local_id
            item_rows[local_id] = {
                "item_local_id": local_id,
                "item_key": item_key,
                "domain": name[1],
                "source_columns": [],
                "source_column_ids": [],
                "delivery_modes": [],
            }

        local_id = item_key_to_local[item_key]
        format_name = assignments.get(name)
        if not format_name or format_name not in formats:
            raise ValueError(f"Missing official SAS format for scored item {name}")

        no_credit, partial, full = _score_codes(formats[format_name])
        if not no_credit or not full:
            raise ValueError(f"Incomplete score definition for {name}: {format_name}")

        existing = item_rows[local_id]
        if existing["source_columns"]:
            if (
                existing["no_credit_codes"] != sorted(no_credit)
                or existing["partial_credit_codes"] != sorted(partial)
                or existing["full_credit_codes"] != sorted(full)
            ):
                raise ValueError(
                    f"Cannot merge {name} into {item_key}: score codes differ"
                )

        delivery_mode = "cba" if name.startswith("C") else "pba"
        source_items.append(
            SourceItem(
                source_name=name,
                source_column_id=source_column_id,
                item_local_id=local_id,
                item_key=item_key,
                label=str(labels[name]),
                delivery_mode=delivery_mode,
                domain=name[1],
                format_name=format_name,
                no_credit_codes=frozenset(no_credit),
                partial_credit_codes=frozenset(partial),
                full_credit_codes=frozenset(full),
            )
        )
        item_rows[local_id]["source_columns"].append(name)
        item_rows[local_id]["source_column_ids"].append(source_column_id)
        item_rows[local_id]["delivery_modes"].append(delivery_mode)
        item_rows[local_id]["format_name"] = format_name
        item_rows[local_id]["no_credit_codes"] = sorted(no_credit)
        item_rows[local_id]["partial_credit_codes"] = sorted(partial)
        item_rows[local_id]["full_credit_codes"] = sorted(full)
        item_rows[local_id]["label"] = str(labels[name])

    for row in item_rows.values():
        row["source_columns"] = "|".join(row["source_columns"])
        row["source_column_ids"] = "|".join(
            str(value) for value in row["source_column_ids"]
        )
        row["delivery_modes"] = "|".join(
            sorted(set(row["delivery_modes"]))
        )

    return source_items, [
        item_rows[index]
        for index in sorted(item_rows)
    ]


def _scope_mask(
    frame: pd.DataFrame,
    scope: AdministrationScope,
) -> pd.Series:
    if scope == "cba":
        return frame["ADMINMODE"].eq(2)
    if scope == "pba":
        return frame["ADMINMODE"].eq(1)
    return frame["ADMINMODE"].isin([1, 2])


def _selected_source_mode(
    frame: pd.DataFrame,
    item: SourceItem,
) -> pd.Series:
    return frame["ADMINMODE"].eq(
        2 if item.delivery_mode == "cba" else 1
    )


def _extract_edges(
    cog_path: Path,
    source_items: list[SourceItem],
    administration_scope: AdministrationScope,
    country: Optional[str],
    chunk_size: int,
    row_limit: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Counter[str]]]:
    id_columns = ["CNT", "CNTSCHID", "CNTSTUID", "ADMINMODE", "BOOKID"]
    usecols = id_columns + [item.source_name for item in source_items]
    student_parts: list[pd.DataFrame] = []
    edge_parts: dict[str, list[np.ndarray]] = {
        "student_ids": [],
        "item_local_ids": [],
        "responses": [],
        "raw_scores": [],
        "source_column_ids": [],
    }
    missing_stats = {
        item.source_name: Counter()
        for item in source_items
    }
    read_kwargs: dict[str, Any] = {
        "chunksize": int(chunk_size),
        "usecols": usecols,
        "user_missing": True,
        "disable_datetime_conversion": True,
    }
    if row_limit > 0:
        read_kwargs["limit"] = int(row_limit)

    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sas7bdat,
        cog_path,
        **read_kwargs,
    )
    normalized_country = country.upper() if country else None

    for frame, _ in reader:
        selected_rows = _scope_mask(frame, administration_scope)
        if normalized_country:
            selected_rows &= (
                frame["CNT"].astype(str).str.upper().eq(normalized_country)
            )
        if not selected_rows.any():
            continue

        student_parts.append(
            frame.loc[selected_rows, id_columns].copy()
        )
        for item in source_items:
            relevant = selected_rows & _selected_source_mode(frame, item)
            if not relevant.any():
                continue

            source = frame[item.source_name]
            for value, count in source.loc[relevant].value_counts(
                dropna=False
            ).items():
                if isinstance(value, str) and value in SPECIAL_MISSING:
                    missing_stats[item.source_name][value] += int(count)
                elif pd.isna(value):
                    missing_stats[item.source_name]["NA"] += int(count)

            numeric = pd.to_numeric(source, errors="coerce")
            unknown = (
                relevant
                & numeric.notna()
                & ~numeric.isin(item.valid_codes)
            )
            if unknown.any():
                values = sorted(numeric.loc[unknown].unique().tolist())
                raise ValueError(
                    f"{item.source_name} has values outside official format: "
                    f"{values}"
                )

            valid = relevant & numeric.isin(item.valid_codes)
            count = int(valid.sum())
            if count == 0:
                continue

            scores = numeric.loc[valid].to_numpy(dtype=np.int16)
            edge_parts["student_ids"].append(
                frame.loc[valid, "CNTSTUID"].to_numpy(dtype=np.int64)
            )
            edge_parts["item_local_ids"].append(
                np.full(count, item.item_local_id, dtype=np.int32)
            )
            edge_parts["responses"].append(
                np.isin(
                    scores.astype(float),
                    list(item.full_credit_codes),
                ).astype(np.uint8)
            )
            edge_parts["raw_scores"].append(scores)
            edge_parts["source_column_ids"].append(
                np.full(count, item.source_column_id, dtype=np.int16)
            )

    if not edge_parts["student_ids"]:
        raise ValueError(
            "No valid scored responses found for the selected configuration"
        )

    edges = {
        key: np.concatenate(parts)
        for key, parts in edge_parts.items()
    }
    students = (
        pd.concat(student_parts, ignore_index=True)
        .drop_duplicates("CNTSTUID")
        .sort_values("CNTSTUID")
    )
    return edges, students, missing_stats


def _attribute_columns(metadata: Any) -> list[str]:
    available = set(metadata.column_names)
    requested = ["CNTSTUID", "W_FSTUWT", "ESCS"]
    requested.extend(
        f"W_FSTURWT{index}"
        for index in range(1, 81)
    )
    for domain in ("MATH", "READ", "SCIE"):
        requested.extend(
            f"PV{index}{domain}"
            for index in range(1, 11)
        )

    return [
        name
        for name in requested
        if name in available
    ]


def _save_student_attributes(
    output_path: Path,
    source_root: Path,
    basic_students: pd.DataFrame,
    selected_student_ids: np.ndarray,
    chunk_size: int,
    include_survey_attributes: bool,
) -> Path:
    base = basic_students.set_index("CNTSTUID").reindex(
        selected_student_ids
    )
    if base.index.has_duplicates:
        raise ValueError("Duplicate CNTSTUID in cognitive attributes")

    if include_survey_attributes:
        qqq_path = _find_named(
            source_root,
            "cy6_ms_cmb_stu_qqq.sas7bdat",
        )
        _, qqq_metadata = pyreadstat.read_sas7bdat(
            qqq_path,
            metadataonly=True,
        )
        columns = _attribute_columns(qqq_metadata)
        qqq_parts: list[pd.DataFrame] = []

        for frame, _ in pyreadstat.read_file_in_chunks(
            pyreadstat.read_sas7bdat,
            qqq_path,
            chunksize=int(chunk_size),
            usecols=columns,
            disable_datetime_conversion=True,
        ):
            keep = frame["CNTSTUID"].isin(selected_student_ids)
            if keep.any():
                qqq_parts.append(frame.loc[keep])

        survey = pd.concat(
            qqq_parts,
            ignore_index=True,
        ).set_index("CNTSTUID")
        if survey.index.has_duplicates:
            raise ValueError(
                "Duplicate CNTSTUID in student questionnaire attributes"
            )
        base = base.join(
            survey.reindex(selected_student_ids),
            how="left",
        )

    arrays: dict[str, np.ndarray] = {
        "student_idx": np.arange(
            len(selected_student_ids),
            dtype=np.int64,
        ),
        "CNTSTUID": selected_student_ids.astype(np.int64),
    }
    for column in base.columns:
        series = base[column]
        if pd.api.types.is_numeric_dtype(series):
            arrays[column] = series.to_numpy()
        else:
            arrays[column] = (
                series.fillna("")
                .astype(str)
                .to_numpy(dtype=str)
            )

    path = output_path.with_suffix(".students.npz")
    np.savez_compressed(path, **arrays)
    return path


def _save_item_and_response_attributes(
    output_path: Path,
    item_rows: list[dict[str, Any]],
    selected_item_local_ids: np.ndarray,
    raw_scores: np.ndarray,
    source_column_ids: np.ndarray,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    selected = [
        item_rows[int(index)]
        for index in selected_item_local_ids
    ]
    for item_idx, row in enumerate(selected):
        row["item_idx"] = item_idx

    item_path = output_path.with_suffix(".items.csv")
    pd.DataFrame(selected).to_csv(item_path, index=False)

    response_path = output_path.with_suffix(".responses.npz")
    np.savez_compressed(
        response_path,
        raw_score=raw_scores,
        source_column_id=source_column_ids,
    )
    return item_path, response_path, selected


def _preprocess_pisa2015(
    input_data_path: str | Path,
    output_path: str | Path,
    *,
    domain: Domain = "full",
    administration_scope: AdministrationScope = "cba",
    save_meta: bool = True,
    min_items_per_student: int = 0,
    min_students_per_item: int = 0,
    country: Optional[str] = None,
    chunk_size: int = 20_000,
    include_survey_attributes: bool = True,
    row_limit: int = 0,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    if domain not in {"R", "M", "S", "full"}:
        raise ValueError("domain must be R, M, S, or full")
    if administration_scope not in {
        "cba",
        "pba",
        "union",
        "intersection",
    }:
        raise ValueError(
            "administration_scope must be cba, pba, union, or intersection"
        )
    if chunk_size <= 0 or row_limit < 0:
        raise ValueError(
            "chunk_size must be positive and row_limit non-negative"
        )

    output = Path(output_path)
    if output.suffix.casefold() != ".pt":
        output = output / f"pisa2015_{administration_scope}_{domain}.pt"

    source_root, cog_path, format_path = _resolve_source(input_data_path)
    source_items, item_rows = _build_item_catalog(
        cog_path,
        format_path,
        domain,
        administration_scope,
    )
    edges, basic_students, missing_stats = _extract_edges(
        cog_path,
        source_items,
        administration_scope,
        country,
        chunk_size,
        row_limit,
    )
    active = _iterative_response_filter(
        edges["student_ids"],
        edges["item_local_ids"],
        min_items_per_student=min_items_per_student,
        min_students_per_item=min_students_per_item,
    )
    indexed = _index_responses(
        edges["student_ids"],
        edges["item_local_ids"],
        edges["responses"],
        edges["raw_scores"],
        edges["source_column_ids"],
        active,
    )
    answers = [1] * len(indexed.item_local_ids)
    _validate_binary_dataset(indexed.data_tensor, answers)

    minimal_metadata: dict[str, Any] = {
        "preprocessor": "pisa_2015",
        "preprocessor_version": PREPROCESSOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_students": len(indexed.student_ids),
        "n_items": len(indexed.item_local_ids),
        "n_choices": 2,
        "n_responses": len(indexed.data_tensor),
        "force_binary": True,
        "strict": True,
        "domains_used": (
            ["R", "M", "S"]
            if domain == "full"
            else [domain]
        ),
        "administration_scope": administration_scope,
    }
    detailed_metadata: dict[str, Any] = {}

    if save_meta:
        output.parent.mkdir(parents=True, exist_ok=True)
        student_path = _save_student_attributes(
            output,
            source_root,
            basic_students,
            indexed.student_ids,
            chunk_size,
            include_survey_attributes,
        )
        item_path, response_path, selected_items = (
            _save_item_and_response_attributes(
                output,
                item_rows,
                indexed.item_local_ids,
                indexed.raw_scores,
                indexed.source_column_ids,
            )
        )
        student_to_idx = {
            str(int(student_id)): index
            for index, student_id in enumerate(indexed.student_ids)
        }
        item_to_idx = {
            str(row["item_key"]): index
            for index, row in enumerate(selected_items)
        }
        detailed_metadata = {
            "student_to_idx": student_to_idx,
            "item_to_idx": item_to_idx,
            "idx_to_student": {
                index: student_id
                for student_id, index in student_to_idx.items()
            },
            "idx_to_item": {
                index: item
                for item, index in item_to_idx.items()
            },
            "source": {
                "root": str(source_root),
                "cognitive_file": str(cog_path),
                "format_file": str(format_path),
            },
            "selection": {
                "domain": domain,
                "administration_scope": administration_scope,
                "country": country,
                "min_items_per_student": min_items_per_student,
                "min_students_per_item": min_students_per_item,
                "row_limit": row_limit,
            },
            "scoring": {
                "binary_one": "official Full credit only",
                "binary_zero": "official Partial credit or No credit",
                "excluded": sorted(SPECIAL_MISSING | {"NA"}),
            },
            "missing_stats": {
                item: dict(counts)
                for item, counts in missing_stats.items()
            },
            "attribute_files": {
                "students": student_path.name,
                "items": item_path.name,
                "responses": response_path.name,
            },
        }

    _save_compatible_cache(
        output,
        indexed.data_tensor,
        answers,
        minimal_metadata,
        detailed_metadata=detailed_metadata,
        save_meta=save_meta,
    )
    returned_metadata = dict(minimal_metadata)
    returned_metadata.update(detailed_metadata)

    return indexed.data_tensor, answers, returned_metadata


def preprocess(**options: Any) -> tuple[ScoredIRTDataset, dict]:
    """
    Preprocess PISA 2015 and adapt its binary output to the scored interface.
    """
    if "output_path" not in options:
        raise TypeError("PISA 2015 preprocessing requires output_path.")

    data, _, metadata = _preprocess_pisa2015(**options)
    return ScoredIRTDataset(data), metadata


def _selected_ids(
    values: Optional[Sequence[int]],
    size: int,
    label: str,
) -> list[int]:
    if values is None:
        return []

    selected = [int(value) for value in values]
    if any(value < 0 or value >= size for value in selected):
        raise IndexError(f"{label} IDs must be between 0 and {size - 1}.")

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
        raise ValueError("artifact_dir is required to locate inspection files.")

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
    """
    Resolve preprocessed indices to PISA source records.

    IDs must refer to the dataset immediately produced by ``preprocess``.
    Experimental subset mappings should be applied before this function.
    """
    students: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    if student_ids is not None:
        path = _artifact_path(metadata, "students", artifact_dir)
        with np.load(path, allow_pickle=False) as attributes:
            if "student_idx" not in attributes:
                raise ValueError("Student attributes do not contain student_idx.")

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
            if any(len(attributes[name]) != n_responses for name in attributes.files):
                raise ValueError("Response attribute lengths are inconsistent.")

            selected = _selected_ids(response_ids, n_responses, "Response")
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
