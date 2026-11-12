"""OpenPOM's molecular graph featurization, ported to NumPy + RDKit.

Mirrors openpom/feat/graph_featurizer.py:
  atoms  -> 134-dim one-hot vector
            (total valence, total degree, total Hs, formal charge,
             atomic number, hybridization, each with an unknown bucket)
  bonds  -> 6-dim vector (dummy 0, single, double, triple, aromatic, in ring)
  each bond appears twice, once per direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from rdkit import Chem

ATOM_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
ATOM_DEGREE_SET = [0, 1, 2, 3, 4, 5]
ATOM_NUM_HS_SET = [0, 1, 2, 3, 4]
ATOM_FORMAL_CHARGE_SET = [-1, -2, 1, 2, 0]  # OpenPOM's exact ordering
ATOM_ATOMIC_NUM_SET = list(range(100))
ATOM_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]

ATOM_FDIM = 134
BOND_FDIM = 6


def _one_hot(value, allowable: Sequence) -> list[float]:
    """One-hot with a trailing unknown bucket, like DeepChem's one_hot_encode."""
    vector = [0.0] * (len(allowable) + 1)
    try:
        vector[allowable.index(value)] = 1.0
    except ValueError:
        vector[-1] = 1.0
    return vector


def atom_features(atom: Chem.Atom) -> list[float]:
    features: list[float] = []
    features += _one_hot(atom.GetTotalValence(), ATOM_VALENCE_SET)
    features += _one_hot(atom.GetTotalDegree(), ATOM_DEGREE_SET)
    features += _one_hot(atom.GetTotalNumHs(), ATOM_NUM_HS_SET)
    features += _one_hot(atom.GetFormalCharge(), ATOM_FORMAL_CHARGE_SET)
    features += _one_hot(atom.GetAtomicNum() - 1, ATOM_ATOMIC_NUM_SET)
    features += _one_hot(str(atom.GetHybridization()), ATOM_HYBRIDIZATION_SET)
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        0.0,
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.IsInRing()),
    ]


@dataclass
class MolGraph:
    """A single featurized molecule."""

    atom_features: np.ndarray  # (N, 134) float32
    edge_index: np.ndarray  # (2, E) int64
    bond_features: np.ndarray  # (E, 6) float32
    smiles: str

    @property
    def num_atoms(self) -> int:
        return int(self.atom_features.shape[0])


def featurize_mol(mol: Chem.Mol, smiles: Optional[str] = None) -> MolGraph:
    atoms = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    src: list[int] = []
    dst: list[int] = []
    bonds: list[list[float]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [begin, end]
        dst += [end, begin]
        b = bond_features(bond)
        bonds.append(b)
        bonds.append(b)

    if src:
        edge_index = np.asarray([src, dst], dtype=np.int64)
        bond_arr = np.asarray(bonds, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        bond_arr = np.zeros((0, BOND_FDIM), dtype=np.float32)

    smiles = smiles if smiles is not None else Chem.MolToSmiles(mol)
    return MolGraph(atoms, edge_index, bond_arr, smiles)


def featurize_smiles(smiles: str) -> Optional[MolGraph]:
    """Featurize one SMILES string; returns None if RDKit cannot parse it."""
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return featurize_mol(mol, smiles=Chem.MolToSmiles(mol))
