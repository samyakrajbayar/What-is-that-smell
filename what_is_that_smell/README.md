# What Is That Smell?

A simulated electronic nose. Pick a scent vial (or paste any molecule as SMILES), start a sniff,
and watch a 16-channel metal-oxide sensor array respond while the trained olfactory model decodes
the odor.

This is the interface half of the project. The model that powers it lives in
[`../smellnet`](../smellnet/README.md).

```
┌───────────────────────┐   ┌────────────────────────────┐   ┌───────────────────────┐
│ 01  SAMPLE RACK       │   │ 02  OLFACTORY CHAMBER      │   │ 03  THE NOSE SAYS     │
│                       │   │                            │   │                       │
│  vanilla   lemon      │   │  molecule structure        │   │  dominant note        │
│  garlic    rose       │   │  + 16 MOS sensor array     │   │  scent fingerprint    │
│  custom SMILES        │   │  + live response traces    │   │  top notes & neighbours│
│  ppm / RH / °C        │   │  + chamber readouts        │   │  closest bottled scent│
└───────────────────────┘   └────────────────────────────┘   └───────────────────────┘
```

## Run it

From the repository root:

```bash
# 1. environment (Python 3.10+; this workspace used 3.13)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. train the nose if you do not have a checkpoint yet
python -m smellnet.train

# 3. start the instrument
python run_app.py
# → http://127.0.0.1:8017
```

The server loads `smellnet/artifacts/model.pt` and `odor_space.npz` at startup. If the checkpoint
is missing it refuses to start with a message telling you to run the trainer.

## Using the instrument

- **Sample rack** — 33 real aroma molecules (vanillin, limonene, eugenol, allicin-adjacent
  disulfides, trimethylamine, ...). Every card sends its SMILES to the model.
- **Custom molecule** — enter any valid SMILES. `Load` validates it with RDKit and shows its
  structure and formula without running a sniff.
- **Concentration** — 1–500 ppm. Response follows a Langmuir-style saturation curve; raising the
  concentration raises the response but with diminishing returns.
- **Humidity / chamber temperature** — 10–95% RH and 15–45 °C. Both modulate the synthetic sensor
  gains, with polar films responding differently from hydrophobic ones.
- **Start sniff** — a 9.2 s cycle: 0.8 s baseline purge, 3.8 s sample draw, then desorption.
  The 16 traces, array heat, chamber phase, peak response and final readout animate in real time.
- **The nose says** — the model's top odor notes, a six-axis odor-family fingerprint, nearest
  neighbours in the 256-d Principal Odor Map embedding, and the closest vial on the rack.

## How the simulation works

A real MOS e-nose measures resistance changes when volatile molecules adsorb on heated metal-oxide
films. The simulation keeps the shape of that process without pretending to be calibrated hardware.

1. **Chemistry** — RDKit computes molecular weight, logP, TPSA, H-bond counts, ring counts and
   functional-group counts (aldehyde, ketone, alcohol, ester, acid, amine, thiol, sulfide, phenol,
   halogen, ...).
2. **Perception** — the MPNN-POM model predicts 138 odor-note probabilities and a 256-d embedding.
   Those probabilities are folded into coarse families (citrus, fruity, floral, sweet, woody, green,
   spicy, savory, animal, chemical, earthy, soapy).
3. **Sensor affinity** — each of the 16 virtual films has a weighted profile over descriptors and
   odor families. A logistic affinity converts the profile match into a 0–1 sensitivity.
4. **Signal** — response amplitude combines affinity, a Langmuir concentration factor
   `c / (c + 18 ppm)`, sensor gain, humidity cross-talk and a small temperature coefficient.
   Fourteen of the channels use tin/tungsten/zinc/indium/iron/chromium/gallium oxide films with
   different selectivities; the array is rounded out by doped variants.
5. **Time** — each channel is a first-order system: it rises toward its amplitude with a
   0.22–0.6 s time constant during exposure and decays with a 0.55–1.4 s constant during purge.
   Deterministic drift and noise (seeded from the molecule, ppm, RH and temperature) make repeat
   runs reproducible.
6. **Identity** — cosine similarity in POM space finds the nearest training molecules and the closest
   bottled scent. If the structure is in the curated reference set, the UI marks it and shows the
   split it belongs to.

The full simulator is in [`sensors.py`](sensors.py) and the odor-family aggregation is in
[`families.py`](families.py).

## API

| Method | Path | Body / notes |
| --- | --- | --- |
| `GET` | `/api/health` | readiness, model and library counts |
| `GET` | `/api/library` | 33 vials, 16 sensor specs, model metrics |
| `POST` | `/api/inspect` | `{"smiles": "..."}` → validated structure, formula, descriptors |
| `POST` | `/api/sniff` | `{"vial_id": "vanilla"}` or `{"smiles": "..."}` plus `ppm`, `humidity`, `temperature` |

`/api/sniff` returns the sample card (with a base64 structure PNG), 16 analysis notes and families,
the full 16 × 116 sensor trace, nearest neighbours and closest rack matches.

## Files

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app, model loading, SMILES validation, structure rendering, neighbours |
| `sensors.py` | 16-channel MOS array definitions and sniff-cycle simulator |
| `families.py` | odor descriptor → family mapping |
| `library.py` | the 33 scent vials |
| `static/index.html` | console layout |
| `static/styles.css` | dark laboratory styling, responsive layout |
| `static/app.js` | sniff animation, canvas traces, radar plot, result rendering |

## Honest notes

- **This is a simulation.** The sensor values are generated; they are not measurements from a
  physical device. The interface and numbers are physically inspired, not calibrated.
- **Odor labels are human descriptors**, not ground-truth chemistry. The model predicts 138
  overlapping notes as independent probabilities, so several notes can be correct at once.
- The structure drawing is RDKit output rendered on the server and cached in memory.
- The app needs the artifacts produced by `smellnet`; it is not a standalone static demo.
