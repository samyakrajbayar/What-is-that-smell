"""A simulated 16-channel metal-oxide (MOS) electronic nose.

Each channel is a virtual chemiresistor with a different catalytic film. Its
affinity is a weighted read of the molecule's physicochemical descriptors plus
the odor families predicted by the neural network. The time trace follows a
first-order adsorption / desorption model with concentration saturation,
humidity cross-talk, drift and noise.

This is a simulation for interaction design, not a hardware calibration.
"""

from __future__ import annotations

import math
import zlib
from typing import Dict, List, Mapping, Sequence

import numpy as np

SAMPLE_RATE_HZ = 12.5
DT = 1.0 / SAMPLE_RATE_HZ
DURATION_S = 9.2
EXPOSURE_START_S = 0.8
EXPOSURE_END_S = 4.6
K_SATURATION_PPM = 18.0

SENSORS: List[dict] = [
    {
        "id": "EN-01", "film": "SnO2", "target": "combustible VOCs", "color": "#5eead4",
        "desc": {"logp": 0.6, "fraction_c": 0.5, "mw": 0.3, "double_bonds": 0.2},
        "family": {"savory": 0.2, "woody": 0.2, "chemical": 0.2},
        "sensitivity": 1.0, "humidity_bias": -0.12,
    },
    {
        "id": "EN-02", "film": "SnO2:Pd", "target": "alcohols & esters", "color": "#fbbf24",
        "desc": {"alcohol": 0.9, "ester": 0.7, "oxygen": 0.5, "tpsa": 0.3},
        "family": {"sweet": 0.35, "fruity": 0.3, "floral": 0.2},
        "sensitivity": 1.08, "humidity_bias": 0.16,
    },
    {
        "id": "EN-03", "film": "WO3", "target": "sulfur compounds", "color": "#f472b6",
        "desc": {"sulfur": 1.0, "thiol": 0.9, "sulfide": 0.9},
        "family": {"savory": 0.35, "chemical": 0.4},
        "sensitivity": 1.22, "humidity_bias": -0.08,
    },
    {
        "id": "EN-04", "film": "ZnO", "target": "aldehydes", "color": "#a3e635",
        "desc": {"aldehyde": 1.0, "oxygen": 0.4, "double_bonds": 0.25},
        "family": {"citrus": 0.3, "sweet": 0.3, "green": 0.2},
        "sensitivity": 1.05, "humidity_bias": -0.04,
    },
    {
        "id": "EN-05", "film": "In2O3", "target": "ketones", "color": "#60a5fa",
        "desc": {"ketone": 0.95, "double_bonds": 0.4, "oxygen": 0.35},
        "family": {"savory": 0.3, "sweet": 0.25},
        "sensitivity": 0.96, "humidity_bias": 0.05,
    },
    {
        "id": "EN-06", "film": "TiO2", "target": "aromatics", "color": "#fb923c",
        "desc": {"aromatic_ring": 0.9, "aromatic": 0.8, "rings": 0.5, "logp": 0.3},
        "family": {"woody": 0.3, "spicy": 0.25, "floral": 0.2},
        "sensitivity": 1.0, "humidity_bias": -0.15,
    },
    {
        "id": "EN-07", "film": "WO3:Au", "target": "terpenes", "color": "#c084fc",
        "desc": {"alkene": 0.8, "rings": 0.5, "mw": 0.3, "logp": 0.3},
        "family": {"green": 0.35, "woody": 0.3, "citrus": 0.2},
        "sensitivity": 1.12, "humidity_bias": -0.18,
    },
    {
        "id": "EN-08", "film": "NiO", "target": "pyrazines & amines", "color": "#34d399",
        "desc": {"nitrogen": 0.8, "amine": 0.7, "rings": 0.4},
        "family": {"savory": 0.35, "animal": 0.35, "earthy": 0.2},
        "sensitivity": 1.02, "humidity_bias": 0.12,
    },
    {
        "id": "EN-09", "film": "SnO2:Pt", "target": "fruit esters", "color": "#e879f9",
        "desc": {"ester": 0.9, "ether": 0.6, "rotatable": 0.4, "mw": 0.2},
        "family": {"fruity": 0.5, "sweet": 0.3, "tropical": 0.2},
        "sensitivity": 1.15, "humidity_bias": 0.03,
    },
    {
        "id": "EN-10", "film": "CuO", "target": "organic acids", "color": "#facc15",
        "desc": {"acid": 1.0, "tpsa": 0.7, "oxygen": 0.4, "hbd": 0.3},
        "family": {"savory": 0.3, "chemical": 0.3, "animal": 0.2},
        "sensitivity": 1.1, "humidity_bias": 0.28,
    },
    {
        "id": "EN-11", "film": "Fe2O3", "target": "phenols", "color": "#38bdf8",
        "desc": {"phenol": 0.95, "aromatic_ring": 0.6, "logp": 0.4},
        "family": {"woody": 0.4, "smoky": 0.2, "spicy": 0.2},
        "sensitivity": 1.08, "humidity_bias": -0.05,
    },
    {
        "id": "EN-12", "film": "Cr2O3", "target": "mercaptans", "color": "#fb7185",
        "desc": {"thiol": 1.0, "sulfur": 0.8, "sulfide": 0.6, "double_bonds": 0.2},
        "family": {"savory": 0.4, "chemical": 0.3, "roasted": 0.2},
        "sensitivity": 1.3, "humidity_bias": -0.1,
    },
    {
        "id": "EN-13", "film": "Ga2O3", "target": "lactones & musks", "color": "#4ade80",
        "desc": {"ester": 0.7, "rings": 0.6, "mw": 0.6, "rotatable": 0.4},
        "family": {"sweet": 0.35, "fruity": 0.3, "creamy": 0.2},
        "sensitivity": 1.18, "humidity_bias": -0.06,
    },
    {
        "id": "EN-14", "film": "ZnO:Al", "target": "bulky hydrophobics", "color": "#f59e0b",
        "desc": {"mw": 0.9, "logp": 0.7, "rotatable": 0.4, "fraction_c": 0.3},
        "family": {"woody": 0.3, "musk": 0.3, "sweet": 0.2},
        "sensitivity": 1.06, "humidity_bias": -0.22,
    },
    {
        "id": "EN-15", "film": "In2O3:Pt", "target": "halogenated VOCs", "color": "#818cf8",
        "desc": {"halogens": 1.0, "logp": 0.5, "tpsa": 0.2},
        "family": {"chemical": 0.4, "solvent": 0.3},
        "sensitivity": 1.14, "humidity_bias": -0.14,
    },
    {
        "id": "EN-16", "film": "SnO2:Ru", "target": "polar & humid", "color": "#2dd4bf",
        "desc": {"tpsa": 0.8, "oxygen": 0.6, "hbd": 0.5, "hba": 0.4},
        "family": {"green": 0.3, "floral": 0.3, "chemical": 0.2},
        "sensitivity": 0.92, "humidity_bias": 0.34,
    },
]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def descriptor_features(descriptors: Mapping[str, float]) -> Dict[str, float]:
    return {
        "logp": _clip((descriptors["logp"] + 1.5) / 8),
        "mw": _clip((descriptors["mw"] - 30) / 250),
        "tpsa": _clip(descriptors["tpsa"] / 100),
        "hbd": _clip(descriptors["hbd"] / 3),
        "hba": _clip(descriptors["hba"] / 5),
        "rings": _clip(descriptors["rings"] / 4),
        "aromatic": _clip(descriptors["aromatic_rings"] / 3),
        "aromatic_ring": _clip(descriptors["aromatic_ring"] / 2),
        "rotatable": _clip(descriptors["rotatable"] / 8),
        "oxygen": _clip(descriptors["oxygen"] / 5),
        "nitrogen": _clip(descriptors["nitrogen"] / 3),
        "sulfur": _clip(descriptors["sulfur"] / 2),
        "halogens": _clip(descriptors["halogens"] / 3),
        "fraction_c": _clip(descriptors["fraction_c"]),
        "double_bonds": _clip(descriptors["double_bonds"] / 5),
        "alcohol": _clip(descriptors["alcohol"] / 2),
        "aldehyde": _clip(descriptors["aldehyde"] / 2),
        "ketone": _clip(descriptors["ketone"] / 2),
        "ester": _clip(descriptors["ester"] / 2),
        "ether": _clip(descriptors["ether"] / 3),
        "acid": _clip(descriptors["acid"] / 2),
        "amine": _clip(descriptors["amine"] / 2),
        "thiol": _clip(descriptors["thiol"]),
        "sulfide": _clip(descriptors["sulfide"] / 2),
        "phenol": _clip(descriptors["phenol"] / 2),
        "alkene": _clip(descriptors["alkene"] / 3),
    }


def _affinity(sensor: dict, features: Mapping[str, float], families: Mapping[str, float]) -> float:
    total, weight = 0.0, 0.0
    for key, coefficient in sensor["desc"].items():
        total += coefficient * features.get(key, 0.0)
        weight += abs(coefficient)
    for key, coefficient in sensor["family"].items():
        total += coefficient * families.get(key, 0.0)
        weight += abs(coefficient)
    weighted = total / max(weight, 1e-6)
    return 1.0 / (1.0 + math.exp(-6.0 * (weighted - 0.30)))


def simulate_array(
    descriptors: Mapping[str, float],
    family_scores: Sequence[Mapping[str, float]],
    ppm: float,
    humidity: float,
    temperature: float,
    seed_key: str,
) -> dict:
    """Simulate a complete sniff cycle for the 16-channel array."""
    features = descriptor_features(descriptors)
    families = {item["family"]: float(item["score"]) for item in family_scores}
    seed = zlib.crc32(
        f"{seed_key}|{ppm:.2f}|{humidity:.1f}|{temperature:.1f}".encode("utf-8")
    )
    rng = np.random.default_rng(seed)

    conc = max(ppm, 0.1)
    concentration_factor = conc / (conc + K_SATURATION_PPM)
    times = np.arange(0.0, DURATION_S + DT / 2, DT)
    humidity_factor = (humidity - 50.0) / 50.0
    temperature_factor = (temperature - 25.0) / 15.0

    channels = []
    for index, sensor in enumerate(SENSORS):
        affinity = _affinity(sensor, features, families)
        polarity = sensor["humidity_bias"]
        condition_gain = (
            1.0
            + polarity * humidity_factor
            + 0.10 * temperature_factor
            + 0.06 * humidity_factor * temperature_factor
        )
        amplitude = (
            (2.2 + 74.0 * concentration_factor * (0.18 + 0.95 * affinity))
            * sensor["sensitivity"]
            * condition_gain
        )
        amplitude = float(np.clip(amplitude, 0.5, 96.0))

        baseline = 1.6 + 0.22 * index + rng.normal(0, 0.25)
        rise_tau = 0.22 + 0.055 * (index % 7) + rng.uniform(0, 0.06)
        decay_tau = 0.55 + 0.09 * ((index * 3) % 8) + rng.uniform(0, 0.15)
        drift_phase = rng.uniform(0, math.tau)
        drift_amp = 0.14 + 0.03 * (index % 4)

        level = 0.0
        readings = []
        for t in times:
            if EXPOSURE_START_S <= t < EXPOSURE_END_S:
                target, tau = amplitude, rise_tau
            else:
                target, tau = 0.0, decay_tau
            level += (target - level) * (DT / tau)
            drift = drift_amp * math.sin(0.55 * t + drift_phase)
            noise = rng.normal(0, 0.12)
            reading = float(np.clip(baseline + level + drift + noise, 0.0, 100.0))
            readings.append(reading)

        peak = float(np.max(readings))
        tail = float(np.mean(readings[-8:]))
        channels.append(
            {
                "id": sensor["id"],
                "film": sensor["film"],
                "target": sensor["target"],
                "color": sensor["color"],
                "affinity": round(affinity, 3),
                "baseline": round(baseline, 2),
                "peak": round(peak, 2),
                "final": round(tail, 2),
                "trace": [round(value, 2) for value in readings],
            }
        )

    return {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "duration_s": DURATION_S,
        "exposure_start_s": EXPOSURE_START_S,
        "exposure_end_s": EXPOSURE_END_S,
        "times": [round(float(t), 2) for t in times],
        "channels": channels,
        "conditions": {
            "ppm": ppm,
            "humidity": humidity,
            "temperature_c": temperature,
            "flow_ml_min": 180,
            "heater_c": 320,
            "chamber_volume_ml": 50,
        },
        "summary": {
            "peak_channel": max(channels, key=lambda channel: channel["peak"])["id"],
            "mean_peak": round(float(np.mean([c["peak"] for c in channels])), 2),
            "max_peak": round(float(np.max([c["peak"] for c in channels])), 2),
        },
    }
