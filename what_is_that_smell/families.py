"""Coarse odor families used for the scent fingerprint and sensor cross-talk."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

FAMILIES: Dict[str, list[str]] = {
    "citrus": ["citrus", "lemon", "orange", "grapefruit", "bergamot", "orangeflower"],
    "fruity": [
        "fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
        "melon", "peach", "pear", "pineapple", "plum", "raspberry",
        "strawberry", "tropical", "juicy", "fruit skin",
    ],
    "floral": [
        "floral", "rose", "jasmin", "lavender", "violet", "lily", "muguet",
        "hyacinth", "hawthorn", "geranium", "orris", "chamomile", "powdery",
    ],
    "sweet": [
        "sweet", "vanilla", "caramellic", "honey", "cocoa", "chocolate",
        "malty", "cream", "creamy", "milky", "buttery", "coconut",
    ],
    "woody": ["woody", "cedar", "sandalwood", "vetiver", "pine", "balsamic", "amber", "musty", "cortex"],
    "green": [
        "green", "grassy", "leafy", "herbal", "fresh", "cucumber", "weedy",
        "hay", "tea", "natural", "clean",
    ],
    "spicy": ["spicy", "cinnamon", "clove", "anisic", "warm", "pungent", "sharp"],
    "savory": [
        "savory", "meaty", "roasted", "cooked", "popcorn", "cheesy",
        "beefy", "malty", "smoky", "burnt", "coffee",
    ],
    "animal": ["animal", "sweaty", "leathery", "fishy", "waxy", "fatty", "oily"],
    "chemical": [
        "chemical", "solvent", "medicinal", "phenolic", "gassy", "sulfurous",
        "alliaceous", "onion", "garlic", "cabbage", "metallic", "ozone",
        "petroleum", "ethereal", "fermented", "sour", "vinegar",
    ],
    "earthy": ["earthy", "mushroom", "musty", "tobacco", "smoky", "hay"],
    "soapy": ["soapy", "clean", "waxy", "powdery", "dry", "odorless"],
}


def family_scores(
    predictions: Iterable[Mapping], tasks: Iterable[str]
) -> list[dict]:
    """Aggregate per-note probabilities into normalized family scores.

    A family's score blends its strongest note with its top-three mean, so a
    single unmistakable note is not diluted by dozens of unrelated notes.
    """
    by_task = {task: float(prob) for task, prob in zip(tasks, predictions)}
    scores: list[dict] = []
    for family, notes in FAMILIES.items():
        values = sorted(
            (by_task[note] for note in notes if note in by_task), reverse=True
        )
        if not values:
            continue
        score = 0.65 * values[0] + 0.35 * (sum(values[:3]) / len(values[:3]))
        scores.append({"family": family, "score": score})
    if not scores:
        return scores
    peak = max(item["score"] for item in scores) or 1.0
    for item in scores:
        item["score"] = item["score"] / peak
    return sorted(scores, key=lambda item: item["score"], reverse=True)
