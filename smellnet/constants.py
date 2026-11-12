"""Odor vocabulary and graph feature constants from OpenPOM.

Reference:
    BioMachineLearning/openpom - Replication of the Principal Odor Map
    (Lee et al., Science 2023).
"""

# The 138 odor descriptors used by the Principal Odor Map / OpenPOM.
TASKS = [
    "alcoholic", "aldehydic", "alliaceous", "almond", "amber", "animal",
    "anisic", "apple", "apricot", "aromatic", "balsamic", "banana", "beefy",
    "bergamot", "berry", "bitter", "black currant", "brandy", "burnt",
    "buttery", "cabbage", "camphoreous", "caramellic", "cedar", "celery",
    "chamomile", "cheesy", "cherry", "chocolate", "cinnamon", "citrus",
    "clean", "clove", "cocoa", "coconut", "coffee", "cognac", "cooked",
    "cooling", "cortex", "coumarinic", "creamy", "cucumber", "dairy", "dry",
    "earthy", "ethereal", "fatty", "fermented", "fishy", "floral", "fresh",
    "fruit skin", "fruity", "garlic", "gassy", "geranium", "grape",
    "grapefruit", "grassy", "green", "hawthorn", "hay", "hazelnut", "herbal",
    "honey", "hyacinth", "jasmin", "juicy", "ketonic", "lactonic", "lavender",
    "leafy", "leathery", "lemon", "lily", "malty", "meaty", "medicinal",
    "melon", "metallic", "milky", "mint", "muguet", "mushroom", "musk",
    "musty", "natural", "nutty", "odorless", "oily", "onion", "orange",
    "orangeflower", "orris", "ozone", "peach", "pear", "phenolic", "pine",
    "pineapple", "plum", "popcorn", "potato", "powdery", "pungent", "radish",
    "raspberry", "ripe", "roasted", "rose", "rummy", "sandalwood", "savory",
    "sharp", "smoky", "soapy", "solvent", "sour", "spicy", "strawberry",
    "sulfurous", "sweaty", "sweet", "tea", "terpenic", "tobacco", "tomato",
    "tropical", "vanilla", "vegetable", "vetiver", "violet", "warm", "waxy",
    "weedy", "winey", "woody",
]

N_TASKS = len(TASKS)  # 138

# Architecture defaults, matching openpom's published ensemble config.
DEFAULT_CONFIG = {
    "n_tasks": N_TASKS,
    "node_out_feats": 100,
    "edge_hidden_feats": 75,
    "edge_out_feats": 100,
    "num_step_message_passing": 5,
    "mpnn_residual": True,
    "message_aggregator_type": "sum",
    "readout_type": "set2set",
    "num_step_set2set": 3,
    "num_layer_set2set": 2,
    "ffn_hidden_list": [392, 392],
    "ffn_embeddings": 256,
    "ffn_activation": "relu",
    "ffn_dropout_p": 0.12,
    "number_atom_features": 134,
    "number_bond_features": 6,
}
