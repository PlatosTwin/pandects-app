"""
Agreement-level split utilities for consistent, leak-free training/evaluation.
"""
# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import random


class SplitMeta(TypedDict):
    val_split: float
    test_split: float
    seed: int
    agreement_col: str
    date_col: str
    year_window: int
    length_bucket_edges: list[float]
    back_matter_bucket_edges: list[float]
    back_label: str


class SplitManifest(TypedDict):
    train: list[str]
    val: list[str]
    test: list[str]
    meta: SplitMeta


def build_agreement_split(
    df: pd.DataFrame,
    *,
    val_split: float,
    test_split: float,
    seed: int = 42,
    agreement_col: str = "agreement_uuid",
    date_col: str = "date_announcement",
    year_window: int = 5,
    length_bucket_edges: list[float] | None = None,
    back_matter_bucket_edges: list[float] | None = None,
    back_label: str = "back_matter",
) -> SplitManifest:
    required_cols = {agreement_col, date_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    announcement_dates = pd.to_datetime(df[date_col], errors="raise")
    if announcement_dates.isna().any():
        raise ValueError(f"Found missing or invalid {date_col} values.")

    working = df[[agreement_col, date_col]].copy()
    working[agreement_col] = working[agreement_col].astype(str)
    if year_window <= 0:
        raise ValueError("year_window must be > 0.")
    working["announcement_year"] = announcement_dates.dt.year
    working["announcement_window"] = (
        (working["announcement_year"] // year_window) * year_window
    )

    year_counts = working.groupby(agreement_col)["announcement_year"].nunique().astype(int)
    inconsistent = year_counts[year_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Found agreements spanning multiple announcement years; cannot stratify by year."
        )

    agreement_ids = working[agreement_col].unique().tolist()
    agreement_windows = cast(
        pd.Series,
        cast(
            object,
            working.drop_duplicates(agreement_col)
            .set_index(agreement_col)
            .loc[agreement_ids, "announcement_window"],
        ),
    )

    def _fixed_buckets(
        values: pd.Series, *, edges: list[float], name: str
    ) -> tuple[pd.Series, list[float]]:
        if len(edges) < 2:
            raise ValueError(f"{name} edges must have at least 2 values.")
        if sorted(edges) != edges:
            raise ValueError(f"{name} edges must be sorted ascending.")
        buckets = pd.cut(values, bins=edges, include_lowest=True, labels=False)
        if buckets.isna().any():
            raise ValueError(f"Failed to bucketize {name} values with edges {edges}.")
        return buckets.astype(int), list(edges)

    if length_bucket_edges is None:
        raise ValueError("length_bucket_edges is required.")
    if back_matter_bucket_edges is None:
        raise ValueError("back_matter_bucket_edges is required.")

    length_edges: list[float] | None = None
    back_edges: list[float] | None = None
    length_bucket = pd.Series(0, index=agreement_ids, dtype=int)
    back_bucket = pd.Series(0, index=agreement_ids, dtype=int)
    back_counts = pd.Series(0, index=agreement_ids, dtype=float)

    page_counts = working.groupby(agreement_col).size().reindex(agreement_ids)
    length_bucket, length_edges = _fixed_buckets(
        page_counts, edges=length_bucket_edges, name="agreement length"
    )
    if "label" not in df.columns:
        raise ValueError("Missing required column for back matter buckets: 'label'.")
    back_counts = (
        df[[agreement_col, "label"]]
        .assign(**{agreement_col: df[agreement_col].astype(str)})
        .groupby(agreement_col)["label"]
        .apply(lambda s: int((s == back_label).sum()))  # pyright: ignore[reportUnknownLambdaType]
        .reindex(agreement_ids)
    )
    back_bucket, back_edges = _fixed_buckets(
        back_counts, edges=back_matter_bucket_edges, name="back matter pages"
    )

    total_split = val_split + test_split
    if total_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    split_names = ["train", "val", "test"]
    split_fracs = {
        "train": 1.0 - total_split,
        "val": val_split,
        "test": test_split,
    }

    def _targets(n: int) -> dict[str, int]:
        raw = {s: n * split_fracs[s] for s in split_names}
        floors = {s: int(np.floor(raw[s])) for s in split_names}
        remainder = n - sum(floors.values())
        if remainder > 0:
            order = sorted(
                split_names,
                key=lambda s: (raw[s] - floors[s]),
                reverse=True,
            )
            for i in range(remainder):
                floors[order[i % len(order)]] += 1
        return floors

    rng = random.Random(seed)
    global_targets = _targets(len(agreement_ids))
    global_counts = {s: 0 for s in split_names}
    back_bucket_counts = {(s, int(b)): 0 for s in split_names for b in back_bucket.unique()}
    back_bucket_sums = {(s, int(b)): 0.0 for s in split_names for b in back_bucket.unique()}
    length_bucket_counts = {
        (s, int(b), int(l)): 0
        for s in split_names
        for b in back_bucket.unique()
        for l in length_bucket.unique()
    }
    year_counts = {
        (s, int(y)): 0 for s in split_names for y in agreement_windows.unique()
    }
    back_bucket_means = back_counts.groupby(back_bucket).mean().to_dict()
    back_bucket_targets = {
        int(b): _targets(int((back_bucket == b).sum()))
        for b in back_bucket.unique()
    }

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    split_lists = {"train": train_ids, "val": val_ids, "test": test_ids}

    for b in sorted(back_bucket.unique()):
        ids_in_back = back_bucket[back_bucket == b].index.tolist()
        preassigned: set[str] = set()
        if back_bucket_targets[int(b)].get("val", 0) == 1 and back_bucket_targets[int(b)].get("test", 0) == 1:
            back_mean = float(back_bucket_means.get(int(b), 0.0))
            ids_sorted = sorted(ids_in_back, key=lambda x: float(back_counts.loc[x]))
            below = [i for i in ids_sorted if float(back_counts.loc[i]) < back_mean]
            above = [i for i in ids_sorted if float(back_counts.loc[i]) > back_mean]
            if below and above:
                val_pick = max(below, key=lambda x: float(back_counts.loc[x]))
                test_pick = min(above, key=lambda x: float(back_counts.loc[x]))
            else:
                closest = sorted(
                    ids_sorted,
                    key=lambda x: abs(float(back_counts.loc[x]) - back_mean),
                )
                picks = closest[:2]
                val_pick = picks[0]
                test_pick = picks[1] if len(picks) > 1 else picks[0]
            for split_name, agr_id in (("val", val_pick), ("test", test_pick)):
                if agr_id in preassigned:
                    continue
                year = int(cast(int, agreement_windows.loc[agr_id]))
                split_lists[split_name].append(agr_id)
                preassigned.add(agr_id)
                global_counts[split_name] += 1
                back_bucket_counts[(split_name, int(b))] += 1
                back_bucket_sums[(split_name, int(b))] += float(
                    cast(float, back_counts.loc[agr_id])
                )
                length_bucket_counts[(split_name, int(b), int(cast(int, length_bucket.loc[agr_id])))] += 1
                year_counts[(split_name, year)] = year_counts.get((split_name, year), 0) + 1
                if back_bucket_targets[int(b)][split_name] > 0:
                    back_bucket_targets[int(b)][split_name] -= 1

        length_subset = cast(pd.Series, length_bucket.loc[ids_in_back])
        by_length = length_subset.groupby(length_subset)
        for l, group in by_length:
            group_ids = [gid for gid in group.index.tolist() if gid not in preassigned]
            rng.shuffle(group_ids)
            for agr_id in group_ids:
                year = int(cast(int, agreement_windows.loc[agr_id]))
                back_value = float(cast(float, back_counts.loc[agr_id]))
                back_mean = float(back_bucket_means.get(int(b), 0.0))
                candidates = [
                    s
                    for s in split_names
                    if back_bucket_targets[int(b)][s] > 0
                ]
                if not candidates:
                    candidates = split_names

                def _score(split_name: str) -> tuple[int, float, int, int, int]:
                    cur_count = back_bucket_counts[(split_name, int(b))]
                    cur_sum = back_bucket_sums[(split_name, int(b))]
                    next_avg = (cur_sum + back_value) / (cur_count + 1)
                    return (
                        length_bucket_counts[(split_name, int(b), int(l))],
                        abs(next_avg - back_mean),
                        year_counts.get((split_name, year), 0),
                        back_bucket_counts[(split_name, int(b))],
                        global_counts[split_name] - global_targets[split_name],
                    )

                chosen = min(candidates, key=_score)
                split_lists[chosen].append(agr_id)
                global_counts[chosen] += 1
                back_bucket_counts[(chosen, int(b))] += 1
                back_bucket_sums[(chosen, int(b))] += back_value
                length_bucket_counts[(chosen, int(b), int(l))] += 1
                year_counts[(chosen, year)] = year_counts.get((chosen, year), 0) + 1
                if back_bucket_targets[int(b)][chosen] > 0:
                    back_bucket_targets[int(b)][chosen] -= 1

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "meta": {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "agreement_col": agreement_col,
            "date_col": date_col,
            "year_window": year_window,
            "length_bucket_edges": length_edges,
            "back_matter_bucket_edges": back_edges,
            "back_label": back_label,
        },
    }


def write_split_manifest(path: str, split: dict[str, object] | SplitManifest) -> None:
    with open(path, "w") as f:
        json.dump(split, f, indent=2, sort_keys=True)


def load_split_manifest(path: str) -> SplitManifest:
    with open(path, "r") as f:
        return cast(SplitManifest, json.load(f))
