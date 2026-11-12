"""Dataset loading, multilabel splitting and featurization for smellnet."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import TASKS
from .featurizer import MolGraph, featurize_smiles

SMILES_COLUMNS = ("nonStereoSMILES", "IsomericSMILES", "smiles", "SMILES")


@dataclass
class OdorDataset:
    smiles: List[str]
    labels: np.ndarray  # (N, 138) float32
    graphs: List[MolGraph] = field(default_factory=list)
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.smiles)


def load_dataset(
    csv_path: str | Path,
    tasks: Sequence[str] = TASKS,
    max_rows: Optional[int] = None,
) -> OdorDataset:
    """Read one of OpenPOM's curated CSVs and featurize its molecules."""
    frame = pd.read_csv(csv_path)
    smiles_column = next((c for c in SMILES_COLUMNS if c in frame.columns), None)
    if smiles_column is None:
        raise ValueError(
            f"{csv_path} has no SMILES column (looked for {SMILES_COLUMNS}); "
            f"found {list(frame.columns[:5])}"
        )
    missing = [task for task in tasks if task not in frame.columns]
    if missing:
        # The Leffingwell source predates a few merged-set descriptors. Keep the
        # shared 138-note head and treat the absent notes as all-negative.
        print(
            f"{csv_path}: filling {len(missing)} absent odor columns with 0 "
            f"({', '.join(missing[:5])}{'…' if len(missing) > 5 else ''})",
            flush=True,
        )
        frame = frame.copy()
        for task in missing:
            frame[task] = 0.0

    if max_rows:
        frame = frame.head(max_rows)

    smiles: List[str] = []
    labels: List[np.ndarray] = []
    graphs: List[MolGraph] = []
    skipped = 0
    for _, row in frame.iterrows():
        raw_smiles = str(row[smiles_column]).strip()
        graph = featurize_smiles(raw_smiles)
        if graph is None:
            skipped += 1
            continue
        smiles.append(graph.smiles)
        labels.append(row[list(tasks)].to_numpy(dtype=np.float32))
        graphs.append(graph)

    if not graphs:
        raise ValueError(f"no parseable molecules found in {csv_path}")
    return OdorDataset(smiles, np.stack(labels), graphs, skipped)


def iterative_stratified_split(
    labels: np.ndarray,
    fractions: Tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> List[np.ndarray]:
    """Multilabel iterative stratification into train/valid/test.

    Uses the proven `iterative-stratification` implementation of the Sechidis
    algorithm. A naive "rarest label first" hand-roll can starve held-out folds
    of notes, so the library does the fold balancing for us. Splits are
    deterministic for a given seed.
    """
    if len(fractions) != 3:
        raise ValueError("this trainer expects three fractions: train/valid/test")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    n = labels.shape[0]
    placeholders = np.zeros((n, 1))
    rest_fraction = fractions[1] + fractions[2]

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=rest_fraction, random_state=seed
    )
    train_idx, rest_idx = next(splitter.split(placeholders, labels))

    valid_share = fractions[1] / rest_fraction
    splitter_rest = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - valid_share, random_state=seed
    )
    valid_rel, test_rel = next(
        splitter_rest.split(np.zeros((len(rest_idx), 1)), labels[rest_idx])
    )
    return [
        train_idx.tolist(),
        rest_idx[valid_rel].tolist(),
        rest_idx[test_rel].tolist(),
    ]


def featurize_all(
    csv_path: str | Path, tasks: Sequence[str] = TASKS, max_rows: Optional[int] = None
) -> OdorDataset:
    start = time.time()
    dataset = load_dataset(csv_path, tasks=tasks, max_rows=max_rows)
    elapsed = time.time() - start
    print(
        f"Featurized {len(dataset)} molecules "
        f"({dataset.skipped} skipped) in {elapsed:.1f}s",
        flush=True,
    )
    return dataset


def compute_note_weights(labels: np.ndarray, mode: str = "inverse") -> np.ndarray:
    """Per-note loss weights over the 138 odor descriptors."""
    positives = labels.sum(axis=0).astype(np.float64)
    if mode == "none":
        weights = np.ones_like(positives)
    elif mode == "openpom":
        weights = np.log1p(positives / positives.max())
    elif mode == "inverse":
        weights = np.log1p(positives.max() / np.maximum(positives, 1.0))
    else:
        raise ValueError(f"unknown loss weighting: {mode}")
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def auc_report(labels: np.ndarray, probs: np.ndarray, tasks: Sequence[str] = TASKS) -> dict:
    """Macro / micro ROC-AUC plus per-task scores (skipping degenerate tasks)."""
    from sklearn.metrics import roc_auc_score

    per_task: dict[str, float] = {}
    valid_scores: List[float] = []
    for index, task in enumerate(tasks):
        target = labels[:, index]
        if target.min() == target.max():
            continue
        score = float(roc_auc_score(target, probs[:, index]))
        per_task[task] = score
        valid_scores.append(score)

    macro = float(np.mean(valid_scores)) if valid_scores else float("nan")
    try:
        micro = float(roc_auc_score(labels.ravel(), probs.ravel()))
    except ValueError:
        micro = float("nan")

    ranking = sorted(per_task.items(), key=lambda item: item[1], reverse=True)
    return {
        "macro_roc_auc": macro,
        "micro_roc_auc": micro,
        "tasks_scored": len(valid_scores),
        "per_task_roc_auc": per_task,
        "strongest_notes": ranking[:10],
        "weakest_notes": ranking[-10:],
    }
