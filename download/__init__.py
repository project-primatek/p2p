"""
Cella Nova Download Module

This module contains scripts for downloading biological data from various sources:
- download_pro.py: Protein sequences from UniProt and STRING
- download_dna.py: DNA sequences and motifs from JASPAR
- download_rna.py: RNA sequences and motifs from ATtRACT
- download_mol.py: Small molecules from ChEMBL and PubChem
"""

from pathlib import Path

# Module directory
MODULE_DIR = Path(__file__).parent

# Default data directory (relative to project root)
DEFAULT_DATA_DIR = MODULE_DIR.parent / "data"

__all__ = [
    "MODULE_DIR",
    "DEFAULT_DATA_DIR",
]
