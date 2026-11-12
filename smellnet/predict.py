"""Predict odor notes for a SMILES string with a trained smellnet checkpoint.

    python -m smellnet.predict "COc1cc(C=O)ccc1O" --top 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .featurizer import featurize_smiles
from .model import collate_graphs, load_model


def predict_smiles(model, smiles: str, top_k: int = 10, device="cpu") -> dict:
    graph = featurize_smiles(smiles)
    if graph is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    batch = collate_graphs([graph]).to(torch.device(device))
    with torch.no_grad():
        proba, _, embedding = model(batch)
    probabilities = proba[0].detach().cpu().numpy()
    tasks = model.tasks if hasattr(model, "tasks") else None
    order = np.argsort(probabilities)[::-1][:top_k]
    return {
        "smiles": graph.smiles,
        "embedding": embedding[0].detach().cpu().numpy(),
        "predictions": [
            {"odor": tasks[i] if tasks else str(i), "probability": float(probabilities[i])}
            for i in order
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", help="molecule to smell")
    parser.add_argument("--checkpoint", default="smellnet/artifacts/model.pt")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. Train one with `python -m smellnet.train`."
        )
    model, metadata = load_model(args.checkpoint)
    model.tasks = metadata["tasks"]
    result = predict_smiles(model, args.smiles, args.top, str(next(model.parameters()).device))

    if args.json:
        print(
            json.dumps(
                {
                    "smiles": result["smiles"],
                    "predictions": result["predictions"],
                },
                indent=2,
            )
        )
        return

    print(f"\nSMILES: {result['smiles']}")
    print("The nose says:\n")
    width = max(len(p["odor"]) for p in result["predictions"])
    for prediction in result["predictions"]:
        bar = "#" * int(round(prediction["probability"] * 40))
        print(f"  {prediction['odor']:<{width}}  {prediction['probability']:.3f}  {bar}")
    print()


if __name__ == "__main__":
    main()
