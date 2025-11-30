"""
Cella Nova Model Package

This package contains neural network models for predicting various
biomolecular interactions:

- model_p2p: Protein-Protein Interaction prediction
- model_p2d: Protein-DNA Interaction prediction
- model_p2r: Protein-RNA Interaction prediction
- model_p2m: Protein-Molecule Interaction prediction
"""

from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = "Cella Nova Team"

# Package directory
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Model weights directory
WEIGHTS_DIR = PACKAGE_DIR

# Default paths for saved models
DEFAULT_MODEL_PATHS = {
    "p2p": WEIGHTS_DIR / "ppi_model.pt",
    "p2p_v2": WEIGHTS_DIR / "ppi_model_v2.pt",
    "p2d": WEIGHTS_DIR / "pdna_model.pt",
    "p2r": WEIGHTS_DIR / "prna_model.pt",
    "p2m": WEIGHTS_DIR / "pmol_model.pt",
}


def get_model_path(model_type: str) -> Path:
    """Get the default path for a model type."""
    if model_type not in DEFAULT_MODEL_PATHS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(DEFAULT_MODEL_PATHS.keys())}"
        )
    return DEFAULT_MODEL_PATHS[model_type]


# Lazy imports to avoid loading heavy dependencies until needed
def __getattr__(name):
    if name == "PPIModel":
        from .model_p2p import PPIModel

        return PPIModel
    elif name == "ProteinDNAModel":
        from .model_p2d import ProteinDNAModel

        return ProteinDNAModel
    elif name == "ProteinRNAModel":
        from .model_p2r import ProteinRNAModel

        return ProteinRNAModel
    elif name == "ProteinMoleculeModel":
        from .model_p2m import ProteinMoleculeModel

        return ProteinMoleculeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PPIModel",
    "ProteinDNAModel",
    "ProteinRNAModel",
    "ProteinMoleculeModel",
    "get_model_path",
    "WEIGHTS_DIR",
    "PROJECT_ROOT",
]
