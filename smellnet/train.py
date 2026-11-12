"""Train the OpenPOM MPNN port on the curated odor datasets.

Example:
    python -m smellnet.train \
        --data openpom-main/openpom-main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv \
        --out smellnet/artifacts --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import TASKS
from .data import (
    auc_report,
    compute_note_weights,
    featurize_all,
    iterative_stratified_split,
)
from .model import build_model, collate_graphs, load_model

DEFAULT_DATA = (
    "openpom-main/openpom-main/openpom/data/curated_datasets/"
    "curated_GS_LF_merged_4983.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="curated OpenPOM CSV")
    parser.add_argument("--out", default="smellnet/artifacts", help="artifact directory")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss-weighting", default="inverse", choices=["inverse", "openpom", "none"]
    )
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=min(8, cores))")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-rows", type=int, default=0, help="debug: cap dataset rows")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, graphs, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(graphs), batch_size):
        batch = collate_graphs(graphs[start : start + batch_size]).to(device)
        proba, _, _ = model(batch)
        outputs.append(proba.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


@torch.no_grad()
def extract_embeddings(model, graphs, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(graphs), batch_size):
        batch = collate_graphs(graphs[start : start + batch_size]).to(device)
        _, _, embeddings = model(batch)
        if embeddings is None:
            raise RuntimeError("model did not return POM embeddings")
        outputs.append(embeddings.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads or min(8, os.cpu_count() or 4))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | threads: {torch.get_num_threads()}", flush=True)
    dataset = featurize_all(args.data, max_rows=args.max_rows or None)
    train_idx, valid_idx, test_idx = iterative_stratified_split(
        dataset.labels, (0.8, 0.1, 0.1), seed=args.seed
    )
    print(
        f"Split: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}",
        flush=True,
    )

    train_graphs = [dataset.graphs[i] for i in train_idx]
    valid_graphs = [dataset.graphs[i] for i in valid_idx]
    test_graphs = [dataset.graphs[i] for i in test_idx]
    train_labels = dataset.labels[train_idx]
    valid_labels = dataset.labels[valid_idx]
    test_labels = dataset.labels[test_idx]

    note_weights_np = compute_note_weights(train_labels, args.loss_weighting)
    note_weights = torch.from_numpy(note_weights_np).to(device)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    rng = np.random.default_rng(args.seed)
    history = []
    best_auc = -1.0
    best_epoch = 0
    started = time.time()
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_graphs))
        total_loss, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            selection = order[start : start + args.batch_size]
            if len(selection) < 2:  # BatchNorm needs more than one sample
                continue
            batch = collate_graphs([train_graphs[i] for i in selection]).to(device)
            target = torch.from_numpy(train_labels[selection]).to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits, _ = model(batch)
            bce = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), target, reduction="none"
            )
            task_loss = (bce * note_weights).sum(dim=1).mean()

            # DeepChem's MPNNPOMModel regularizer: L1 + L2 on non-bias weights.
            l1 = torch.zeros((), device=device)
            l2 = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                if "bias" not in name and param.ndim > 1:
                    l1 = l1 + torch.norm(param, p=1)
                    l2 = l2 + torch.norm(param, p=2)
            loss = task_loss + args.weight_decay * (l1 + l2)

            loss.backward()
            optimizer.step()
            total_loss += float(task_loss.detach())
            batches += 1

        train_loss = total_loss / max(batches, 1)
        valid_probs = predict(model, valid_graphs, args.batch_size, device)
        valid_auc = auc_report(valid_labels, valid_probs)["macro_roc_auc"]
        scheduler.step()
        epochs_run = epoch
        elapsed = time.time() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_macro_roc_auc": valid_auc,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
            f"valid AUC {valid_auc:.4f}  lr {optimizer.param_groups[0]['lr']:.5f}  "
            f"{elapsed:.0f}s",
            flush=True,
        )

        if valid_auc > best_auc + 1e-5:
            best_auc = valid_auc
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "tasks": TASKS,
                    "epoch": epoch,
                    "valid_macro_roc_auc": valid_auc,
                },
                out_dir / "model.pt",
            )
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best {best_epoch}).", flush=True)
            break

    total_seconds = time.time() - started
    model, meta = load_model(out_dir / "model.pt", device)

    train_probs = predict(model, train_graphs, 256, device)
    valid_probs = predict(model, valid_graphs, 256, device)
    test_probs = predict(model, test_graphs, 256, device)
    train_report = auc_report(train_labels, train_probs)
    valid_report = auc_report(valid_labels, valid_probs)
    test_report = auc_report(test_labels, test_probs)

    print(
        f"Best epoch {best_epoch} | test macro ROC-AUC "
        f"{test_report['macro_roc_auc']:.4f} | micro {test_report['micro_roc_auc']:.4f}",
        flush=True,
    )

    metrics = {
        "model": "MPNNPOM (pure-PyTorch port of OpenPOM)",
        "dataset": {
            "path": str(args.data),
            "molecules": len(dataset),
            "tasks": len(TASKS),
            "skipped": dataset.skipped,
            "positive_labels_per_note_min": float(dataset.labels.sum(axis=0).min()),
            "positive_labels_per_note_max": float(dataset.labels.sum(axis=0).max()),
        },
        "split": {
            "train": len(train_idx),
            "valid": len(valid_idx),
            "test": len(test_idx),
            "seed": args.seed,
            "strategy": "multilabel iterative stratification (Sechidis)",
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_valid_macro_roc_auc": best_auc,
            "seconds": round(total_seconds, 1),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "loss_weighting": args.loss_weighting,
            "parameters": n_params,
            "device": str(device),
        },
        "train": train_report,
        "valid": valid_report,
        "test": test_report,
        "checkpoint": {
            "epoch": int(meta.get("epoch", best_epoch)),
            "valid_macro_roc_auc": float(meta.get("valid_macro_roc_auc", best_auc)),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    # Odor embedding library: every molecule's POM vector + labels for the app.
    embeddings = extract_embeddings(model, dataset.graphs, 256, device)
    split_ids = np.zeros(len(dataset), dtype=np.int8)
    split_ids[valid_idx] = 1
    split_ids[test_idx] = 2
    np.savez_compressed(
        out_dir / "odor_space.npz",
        smiles=np.asarray(dataset.smiles, dtype=str),
        labels=dataset.labels.astype(np.float32),
        embeddings=embeddings.astype(np.float32),
        split=split_ids,
        tasks=np.asarray(TASKS, dtype=str),
    )
    print(f"Saved artifacts to {out_dir}", flush=True)

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [h["epoch"] for h in history]
            losses = [h["train_loss"] for h in history]
            aucs = [h["valid_macro_roc_auc"] for h in history]
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.plot(epochs, losses, color="#d97706", linewidth=2, label="train loss")
            axis.set_xlabel("epoch")
            axis.set_ylabel("weighted BCE loss", color="#d97706")
            axis.tick_params(axis="y", labelcolor="#d97706")
            twin = axis.twinx()
            twin.plot(epochs, aucs, color="#0f766e", linewidth=2, label="valid macro ROC-AUC")
            twin.set_ylabel("valid macro ROC-AUC", color="#0f766e")
            twin.tick_params(axis="y", labelcolor="#0f766e")
            axis.set_title("SmellNet training on OpenPOM curated odor data")
            figure.tight_layout()
            figure.savefig(out_dir / "training_curves.png", dpi=140)
            plt.close(figure)
        except Exception as error:  # plotting is a convenience, not a dependency
            print(f"Could not draw training curves: {error}", flush=True)


if __name__ == "__main__":
    main()
