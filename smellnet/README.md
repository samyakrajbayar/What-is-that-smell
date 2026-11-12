# SmellNet — an AI that can smell

SmellNet trains a graph neural network to map a molecule's structure to **138 human odor
descriptors**. It is a faithful port of the MPNN-POM model from
[OpenPOM](https://github.com/BioMachineLearning/openpom) (a replication of the Principal Odor Map,
Lee et al., *Science* 2023), retrained on the curated datasets shipped with that repository.

The trained checkpoint, POM embeddings and metrics feed the
[What Is That Smell?](../what_is_that_smell/README.md) electronic-nose app.

## Why a port?

OpenPOM's reference implementation depends on DeepChem + DGL. This workspace runs Windows with
Python 3.13, where a compatible DGL/DeepChem stack is not available (the PyPI `dgl` name is an
unrelated abandoned package). The model below reproduces OpenPOM's layer math in plain PyTorch, so
it runs on CPU anywhere, with only PyTorch + RDKit + NumPy.

Two details differ from the reference implementation, both documented in the code:

- **Loss aggregation** — OpenPOM's `CustomMultiLabelLoss` feeds `[1 - p, p]` to cross-entropy. This
  port uses the intended equivalent, binary cross-entropy on the logits, with the same per-note
  weighting (default: inverse-frequency `log1p(max_count / count)`, normalized to mean 1). Use
  `--loss-weighting openpom` to use OpenPOM's `log1p(count / max_count)` variant.
- **No 10-model ensemble** — the repository's `models/ensemble_models/*/checkpoint2.pt` files are
  133-byte Git LFS pointers, not weights. This project trains one model from the bundled data.

## Architecture

```
SMILES
  └─ RDKit graph: 134-dim atom one-hots, 6-dim bond vectors, both bond directions
       └─ Linear(134→100) + ReLU
            └─ 5 × [ NNConv(100→100, edge-conditioned, residual) → ReLU → GRU(100→100) ]
                 └─ fold atoms + projected bond embeddings (100+100)
                      └─ Set2Set(200, 3 iterations, 2 layers)   → 400-d graph vector
                           └─ FFN 400→392→392→256 (BatchNorm, dropout 0.12)
                                ├─ 138 odor logits → sigmoid probabilities
                                └─ 256-d Principal Odor Map embedding
```

- **Featurization** matches `openpom/feat/graph_featurizer.py`: valence, degree, total Hs, formal
  charge, atomic number (with OpenPOM's `atomic_num - 1` indexing) and hybridization, each with an
  unknown bucket; bonds are `[dummy, single, double, triple, aromatic, in_ring]`.
- **Message passing** matches DGL `NNConv(residual=True)`: `m_ij = h_j W(e_ij)`, summed at the
  destination, then residual and bias; the GRU consumes each node independently.
- **Set2Set** follows DGL's implementation, including the LSTM initialization and per-graph softmax.
- 2,088,027 parameters. The model returns probabilities, logits and the 256-d embedding.

## Dataset

`openpom-main/openpom-main/openpom/data/curated_datasets/`

| File | Rows | Notes |
| --- | --- | --- |
| `curated_GS_LF_merged_4983.csv` | 4,983 | merged GoodScents + Leffingwell, canonical training set |
| `curated_goodcents.csv` | 4,392 | source dataset (also supported with `--data`) |
| `curated_leffingwell.csv` | 3,510 | source dataset; lacks `cortex`, filled with zeros when used |

All rows parse; there are no duplicate SMILES in the merged set. Labels range from 31 to 1,902
positives per note. Splits use deterministic multilabel iterative stratification (Sechidis) via the
`iterative-stratification` package with seed 42 and an 80/10/10 split. All 138
notes are represented in train, validation and test.

## Train

From the repository root:

```bash
python -m smellnet.train
# or be explicit
python -m smellnet.train \
  --data openpom-main/openpom-main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv \
  --out smellnet/artifacts \
  --epochs 32 --batch-size 128 --lr 1e-3 --patience 8 --seed 42
```

Training uses Adam, a StepLR that halves the learning rate every 15 epochs, and DeepChem-style
L1+L2 regularization (`1e-5` on non-bias weights). The best validation macro ROC-AUC checkpoint is
kept. On this 8-core CPU, one epoch takes roughly 45 s; the recorded run completed
32 epochs in about 23 minutes.

Useful options:

| Flag | Meaning |
| --- | --- |
| `--max-rows N` | quick smoke test on N molecules |
| `--loss-weighting inverse \| openpom \| none` | rare-note weighting scheme |
| `--threads 8` | CPU threads (defaults to `min(8, cores)`) |
| `--device cuda` | train on a GPU if available |
| `--no-plot` | skip the training-curve PNG |

## Evaluate and predict

Training writes a report to `artifacts/metrics.json` (macro/micro ROC-AUC, per-note scores, split
sizes, timings, parameters) and curves to `artifacts/training_curves.png`.

```bash
python -m smellnet.predict "COc1cc(C=O)ccc1O" --top 8
python -m smellnet.predict "COc1cc(C=O)ccc1O" --json
```

The CLI prints the highest-probability notes. Attach `--checkpoint path/to/model.pt` to use another
run.

## Artifacts

| File | Contents |
| --- | --- |
| `model.pt` | state dict + architecture config + 138-note vocabulary + best epoch/score |
| `metrics.json` | dataset stats, split, test/valid/train metrics, hyperparameters |
| `training_history.json` | loss/AUC/learning-rate per epoch |
| `training_curves.png` | loss and validation AUC |
| `odor_space.npz` | canonical SMILES, 138 labels, 256-d embeddings, train/valid/test flags |

## Notes and limitations

- A single model is not the paper's 10-model × 5-fold ensemble, so numbers are not directly
  comparable to the published 0.8872 ROC-AUC.
- Macro ROC-AUC over 138 tasks is noisy because rare notes have few test positives; per-note scores
  are in `metrics.json`.
- Odor descriptors are cultural/perceptual labels from perfume-industry data, not chemical truths.
- The model predicts probabilities, not certainty. The app surfaces the top notes, not a single
  ontological answer.

## References

- Lee, B. K. et al. *A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception.*
  Science 381, 999–1006 (2023). <https://doi.org/10.1126/science.ade4401>
- OpenPOM: <https://github.com/BioMachineLearning/openpom>
- DGL NNConv and Set2Set implementations, used as behavioral references.
