"""Molecular descriptors used by the simulated electronic nose."""

from __future__ import annotations

from typing import Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

_SMARTS = {
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H]",
    "phenol": "c[OX2H]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "ether": "[OD2]([#6])[#6]",
    "acid": "[CX3](=O)[OX2H1]",
    "amine": "[NX3;!$(NC=O)]",  # primary, secondary and tertiary amines
    "thiol": "[SX2H]",
    "sulfide": "[#16X2H0]",
    "nitrile": "[NX1]#[CX2]",
    "alkene": "[CX3]=[CX3]",
    "aromatic_ring": "a1aaaaa1",
}
_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in _SMARTS.items()}


def describe_mol(mol: Chem.Mol) -> Dict[str, float]:
    """Physicochemical descriptors from a parsed RDKit molecule."""
    total_atoms = max(mol.GetNumAtoms(), 1)
    counts = {
        "C": 0, "O": 0, "N": 0, "S": 0, "P": 0, "F": 0, "Cl": 0, "Br": 0, "I": 0,
    }
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in counts:
            counts[symbol] += 1

    functional = {
        name: len(mol.GetSubstructMatches(pattern)) if pattern is not None else 0
        for name, pattern in _PATTERNS.items()
    }

    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rotatable": float(Lipinski.NumRotatableBonds(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "fraction_c": counts["C"] / total_atoms,
        "oxygen": float(counts["O"]),
        "nitrogen": float(counts["N"]),
        "sulfur": float(counts["S"]),
        "phosphorus": float(counts["P"]),
        "halogens": float(counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]),
        "double_bonds": float(sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0)),
        **{name: float(value) for name, value in functional.items()},
    }


def describe_smiles(smiles: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return describe_mol(mol)
