"""SmellNet - an open olfactory model trained on OpenPOM's curated odor data.

The network is a faithful, dependency-light PyTorch port of OpenPOM's
MPNN-POM (BioMachineLearning/openpom): the same 134-dim atom / 6-dim bond
featurization, NNConv message passing with GRU updates, Set2Set readout and
feed-forward principal odor map head. It runs without DGL or DeepChem.
"""

from .constants import TASKS, N_TASKS

__version__ = "1.0.0"
__all__ = ["TASKS", "N_TASKS", "__version__"]
