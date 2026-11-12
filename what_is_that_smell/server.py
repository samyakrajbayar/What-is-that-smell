"""FastAPI backend for the What Is That Smell? simulated electronic nose."""

from __future__ import annotations

import base64
import io
import json
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Draw

from smellnet.constants import TASKS
from smellnet.descriptors import describe_mol
from smellnet.featurizer import featurize_mol, featurize_smiles
from smellnet.model import collate_graphs, load_model

from .families import family_scores
from .library import LIBRARY
from .sensors import SENSORS, simulate_array

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
ARTIFACT_DIR = ROOT / "smellnet" / "artifacts"


@dataclass
class NoseState:
    model: Optional[torch.nn.Module] = None
    tasks: List[str] = field(default_factory=lambda: list(TASKS))
    metrics: dict = field(default_factory=dict)
    space_smiles: Optional[np.ndarray] = None
    space_embeddings: Optional[np.ndarray] = None
    space_split: Optional[np.ndarray] = None
    canonical_lookup: Dict[str, int] = field(default_factory=dict)
    library: List[dict] = field(default_factory=list)
    library_embeddings: Optional[np.ndarray] = None
    structure_cache: Dict[str, str] = field(default_factory=dict)


STATE = NoseState()
DEVICE = torch.device("cpu")


def _embed_graph(model, graph) -> np.ndarray:
    batch = collate_graphs([graph])
    with torch.no_grad():
        _, _, embedding = model(batch)
    vector = embedding[0].detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def load_resources() -> None:
    checkpoint = ARTIFACT_DIR / "model.pt"
    space_path = ARTIFACT_DIR / "odor_space.npz"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not checkpoint.exists():
        raise RuntimeError(
            f"No trained model at {checkpoint}. Run `python -m smellnet.train` first."
        )

    model, metadata = load_model(checkpoint, DEVICE)
    model.tasks = list(metadata.get("tasks", TASKS))
    STATE.model = model
    STATE.tasks = model.tasks
    if metrics_path.exists():
        STATE.metrics = json.loads(metrics_path.read_text())

    if space_path.exists():
        space = np.load(space_path, allow_pickle=False)
        STATE.space_smiles = space["smiles"]
        STATE.space_embeddings = _normalize_rows(space["embeddings"].astype(np.float32))
        STATE.space_split = space["split"]
        # Training stored canonical SMILES, so the lookup needs no re-parsing.
        STATE.canonical_lookup = {
            str(smiles): index for index, smiles in enumerate(STATE.space_smiles)
        }

    valid_library: List[dict] = []
    embeddings: List[np.ndarray] = []
    for vial in LIBRARY:
        graph = featurize_smiles(vial["smiles"])
        if graph is None:
            print(f"Skipping invalid vial SMILES: {vial['name']} {vial['smiles']}")
            continue
        mol = Chem.MolFromSmiles(vial["smiles"])
        descriptors = describe_mol(mol)
        reference_index = STATE.canonical_lookup.get(graph.smiles)
        reference_split = None
        if reference_index is not None and STATE.space_split is not None:
            reference_split = {0: "train", 1: "valid", 2: "test"}.get(
                int(STATE.space_split[reference_index]), "train"
            )
        valid_library.append(
            {
                **vial,
                "canonical_smiles": graph.smiles,
                "formula": descriptors["formula"],
                "mw": round(descriptors["mw"], 1),
                "in_reference": reference_index is not None,
                "reference_split": reference_split,
                "structure_png": _structure_png(graph.smiles),
            }
        )
        embeddings.append(_embed_graph(model, graph))
    STATE.library = valid_library
    STATE.library_embeddings = (
        _normalize_rows(np.stack(embeddings)) if embeddings else None
    )
    print(
        f"Nose ready: {len(valid_library)} vials, {len(STATE.canonical_lookup)} "
        f"reference molecules, {len(STATE.tasks)} odor notes"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="What Is That Smell?",
    description="A simulated 16-channel electronic nose driven by an OpenPOM-based odor model.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SniffRequest(BaseModel):
    vial_id: Optional[str] = None
    smiles: Optional[str] = None
    ppm: float = Field(25.0, ge=1.0, le=500.0)
    humidity: float = Field(50.0, ge=10.0, le=95.0)
    temperature: float = Field(25.0, ge=15.0, le=45.0)


def _structure_png(smiles: str) -> str:
    cached = STATE.structure_cache.get(smiles)
    if cached:
        return cached
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    image = Draw.MolToImage(mol, size=(460, 340), kekulize=True)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    if len(STATE.structure_cache) < 256:
        STATE.structure_cache[smiles] = url
    return url


def _neighbors(embedding: np.ndarray, exclude_smiles: str, limit: int = 4) -> List[dict]:
    if STATE.space_embeddings is None or STATE.space_smiles is None:
        return []
    scores = STATE.space_embeddings @ embedding
    library_by_smiles = {
        vial["canonical_smiles"]: vial["name"] for vial in STATE.library
    }
    order = np.argsort(scores)[::-1]
    neighbors: List[dict] = []
    split_names = {0: "train", 1: "valid", 2: "test"}
    for index in order:
        smiles = str(STATE.space_smiles[index])
        if smiles == exclude_smiles:
            continue
        split_id = int(STATE.space_split[index]) if STATE.space_split is not None else 0
        neighbors.append(
            {
                "smiles": smiles,
                "name": library_by_smiles.get(smiles),
                "similarity": round(float(scores[index]), 4),
                "split": split_names.get(split_id, "train"),
            }
        )
        if len(neighbors) >= limit:
            break
    return neighbors


def _closest_vials(embedding: np.ndarray, exclude_smiles: str, limit: int = 3) -> List[dict]:
    if STATE.library_embeddings is None:
        return []
    scores = STATE.library_embeddings @ embedding
    order = np.argsort(scores)[::-1]
    matches = []
    for index in order:
        vial = STATE.library[int(index)]
        if vial["canonical_smiles"] == exclude_smiles:
            continue
        matches.append(
            {
                "name": vial["name"],
                "compound": vial["compound"],
                "family": vial["family"],
                "similarity": round(float(scores[index]), 4),
            }
        )
        if len(matches) >= limit:
            break
    return matches


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if STATE.model is not None else "loading",
        "model_loaded": STATE.model is not None,
        "vials": len(STATE.library),
        "notes": len(STATE.tasks),
    }


@app.get("/api/library")
def library() -> dict:
    test_auc = STATE.metrics.get("test", {}).get("macro_roc_auc") if STATE.metrics else None
    return {
        "vials": STATE.library,
        "model": {
            "name": "MPNN-POM",
            "architecture": "OpenPOM MPNN (PyTorch port)",
            "notes": len(STATE.tasks),
            "molecules": STATE.metrics.get("dataset", {}).get("molecules"),
            "test_macro_roc_auc": test_auc,
            "trained_epochs": STATE.metrics.get("training", {}).get("epochs_run"),
            "best_epoch": STATE.metrics.get("training", {}).get("best_epoch"),
        },
        "sensors": [
            {"id": s["id"], "film": s["film"], "target": s["target"], "color": s["color"]}
            for s in SENSORS
        ],
        "families": sorted({vial["family"] for vial in STATE.library}),
    }


class InspectRequest(BaseModel):
    smiles: str


@app.post("/api/inspect")
def inspect(request: InspectRequest) -> dict:
    """Validate a custom SMILES and return its identity card, without sniffing."""
    raw = request.smiles.strip()
    if not raw:
        raise HTTPException(status_code=422, detail="Enter a SMILES string first.")
    if len(raw) > 300:
        raise HTTPException(status_code=422, detail="That SMILES string is too long.")
    mol = Chem.MolFromSmiles(raw)
    if mol is None or mol.GetNumAtoms() == 0:
        raise HTTPException(status_code=422, detail="RDKit could not parse that SMILES string.")
    canonical = Chem.MolToSmiles(mol)
    descriptors = describe_mol(mol)
    known_index = STATE.canonical_lookup.get(canonical)
    known_split = None
    if known_index is not None and STATE.space_split is not None:
        known_split = {0: "train", 1: "valid", 2: "test"}.get(
            int(STATE.space_split[known_index]), "train"
        )
    return {
        "name": "Unknown compound",
        "compound": "custom molecule",
        "family": "",
        "smiles": canonical,
        "formula": descriptors["formula"],
        "mw": round(descriptors["mw"], 2),
        "logp": round(descriptors["logp"], 2),
        "tpsa": round(descriptors["tpsa"], 1),
        "heavy_atoms": int(descriptors["heavy_atoms"]),
        "in_reference": known_index is not None,
        "reference_split": known_split,
        "structure_png": _structure_png(canonical),
    }


@app.post("/api/sniff")
def sniff(request: SniffRequest) -> dict:
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="The nose is still warming up.")

    vial = next((v for v in STATE.library if v["id"] == request.vial_id), None)
    raw_smiles = vial["smiles"] if vial else (request.smiles or "")
    raw_smiles = raw_smiles.strip()
    if not raw_smiles:
        raise HTTPException(status_code=422, detail="Choose a vial or enter a SMILES string.")
    if len(raw_smiles) > 300:
        raise HTTPException(status_code=422, detail="That SMILES string is too long.")

    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise HTTPException(status_code=422, detail="RDKit could not parse that SMILES string.")
    graph = featurize_mol(mol, smiles=Chem.MolToSmiles(mol))
    descriptors = describe_mol(mol)

    batch = collate_graphs([graph])
    with torch.no_grad():
        probabilities, _, embedding = STATE.model(batch)
    probs = probabilities[0].detach().cpu().numpy()
    emb = embedding[0].detach().cpu().numpy().astype(np.float32)
    emb = emb / (float(np.linalg.norm(emb)) or 1.0)

    ranked = np.argsort(probs)[::-1]
    top_notes = [
        {"odor": STATE.tasks[int(index)], "probability": round(float(probs[index]), 4)}
        for index in ranked[:16]
    ]
    detected = [
        {"odor": STATE.tasks[int(index)], "probability": round(float(probs[index]), 4)}
        for index in ranked
        if float(probs[index]) >= 0.25
    ][:8]
    if not detected:
        detected = top_notes[:3]

    families = family_scores(probs.tolist(), STATE.tasks)
    families = [
        {"family": item["family"], "score": round(float(item["score"]), 4)}
        for item in families[:6]
    ]

    sensors = simulate_array(
        descriptors,
        families,
        request.ppm,
        request.humidity,
        request.temperature,
        seed_key=graph.smiles,
    )

    known_index = STATE.canonical_lookup.get(graph.smiles)
    known = known_index is not None
    if known and STATE.space_split is not None:
        known_split = {0: "train", 1: "valid", 2: "test"}.get(
            int(STATE.space_split[known_index]), "train"
        )
    else:
        known_split = None

    return {
        "sample": {
            "name": vial["name"] if vial else "Unknown compound",
            "compound": vial["compound"] if vial else "",
            "family": vial["family"] if vial else "",
            "smiles": graph.smiles,
            "formula": descriptors["formula"],
            "mw": round(descriptors["mw"], 2),
            "logp": round(descriptors["logp"], 2),
            "tpsa": round(descriptors["tpsa"], 1),
            "heavy_atoms": int(descriptors["heavy_atoms"]),
            "in_reference": known,
            "reference_split": known_split,
            "structure_png": _structure_png(graph.smiles),
        },
        "analysis": {
            "top_notes": top_notes,
            "detected_notes": detected,
            "families": families,
            "verdict": top_notes[0]["odor"],
            "confidence": top_notes[0]["probability"],
        },
        "sensors": sensors,
        "neighbors": _neighbors(emb, graph.smiles),
        "closest_vials": _closest_vials(emb, graph.smiles),
        "model": {
            "test_macro_roc_auc": STATE.metrics.get("test", {}).get("macro_roc_auc")
            if STATE.metrics
            else None,
            "notes": len(STATE.tasks),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8017")),
    )
