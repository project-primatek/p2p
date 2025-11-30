#!/usr/bin/env python3
"""
Prepare Protein-Molecule Interaction (P2M) Data for Training

This script processes raw downloaded protein-molecule interaction data and prepares it for
training the P2M interaction prediction model.

Steps:
1. Load protein sequences from UniProt downloads
2. Load molecule data (SMILES) from ChEMBL/PubChem
3. Load protein-molecule interaction pairs with binding affinities
4. Filter and validate sequences/molecules
5. Generate negative samples using intelligent sampling strategies
6. Split data into train/val/test sets
7. Compute sequence features and statistics
8. Save prepared data in training-ready format
"""

import argparse
import gzip
import json
import logging
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Determine project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared" / "p2m"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# SMILES character set for validation
SMILES_CHARS = set("CNOPSFIBrcnopsfi[]()=#@+-.0123456789%\\/lHaeKgZdVuytMwACGTb")


@dataclass
class ProteinInfo:
    """Store protein information"""

    uniprot_id: str
    sequence: str
    gene_name: Optional[str] = None
    description: Optional[str] = None
    target_type: Optional[str] = None  # e.g., "enzyme", "receptor", etc.
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class MoleculeInfo:
    """Store molecule information"""

    molecule_id: str
    smiles: str
    name: Optional[str] = None
    molecular_weight: Optional[float] = None
    source: str = "ChEMBL"

    @property
    def length(self) -> int:
        return len(self.smiles)


@dataclass
class InteractionPair:
    """Store protein-molecule interaction pair information"""

    protein_id: str
    protein_seq: str
    smiles: str
    label: int  # 1 for positive (binding), 0 for negative (non-binding)
    affinity: float = 0.0  # Binding affinity (e.g., pIC50, pKd)
    source: str = "ChEMBL"
    activity_type: Optional[str] = None  # e.g., "IC50", "Ki", "Kd"


@dataclass
class DataSplit:
    """Store data split information"""

    train_pairs: List[InteractionPair]
    val_pairs: List[InteractionPair]
    test_pairs: List[InteractionPair]

    @property
    def total_pairs(self) -> int:
        return len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)


class PMolDataPreparer:
    """Prepare protein-molecule interaction data for training"""

    # Standard amino acids
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        min_protein_length: int = 20,
        max_protein_length: int = 2000,
        min_smiles_length: int = 3,
        max_smiles_length: int = 200,
        affinity_threshold: float = 5.0,  # pIC50/pKd threshold for positive
        negative_ratio: float = 1.0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_protein_length = min_protein_length
        self.max_protein_length = max_protein_length
        self.min_smiles_length = min_smiles_length
        self.max_smiles_length = max_smiles_length
        self.affinity_threshold = affinity_threshold
        self.negative_ratio = negative_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Data containers
        self.proteins: Dict[str, ProteinInfo] = {}
        self.molecules: Dict[str, MoleculeInfo] = {}
        self.interactions: List[InteractionPair] = []
        self.protein_mol_pairs: Set[Tuple[str, str]] = set()

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_proteins(self) -> int:
        """Load protein sequences from various sources"""
        logger.info("Loading protein sequences...")

        total_loaded = 0

        # Load from FASTA files
        fasta_patterns = [
            "**/*proteins*.fasta",
            "**/*targets*.fasta",
            "**/uniprot/*.fasta",
            "**/*.fasta",
        ]

        for pattern in fasta_patterns:
            for fasta_file in self.data_dir.glob(pattern):
                count = self._load_fasta(fasta_file)
                total_loaded += count

        # Load from JSON files
        for json_file in self.data_dir.glob("**/*proteins*.json"):
            count = self._load_json_proteins(json_file)
            total_loaded += count

        # Load from target files
        for target_file in self.data_dir.glob("**/*targets*.json"):
            count = self._load_json_proteins(target_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.proteins):,} unique proteins")
        return len(self.proteins)

    def _load_fasta(self, filepath: Path) -> int:
        """Load sequences from FASTA file"""
        count = 0

        try:
            open_func = gzip.open if filepath.suffix == ".gz" else open
            mode = "rt" if filepath.suffix == ".gz" else "r"

            with open_func(filepath, mode) as f:
                current_id = None
                current_seq = []
                current_desc = None

                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        # Save previous protein
                        if current_id and current_seq:
                            seq = "".join(current_seq)
                            if self._is_valid_protein(seq):
                                protein = ProteinInfo(
                                    uniprot_id=current_id,
                                    sequence=seq,
                                    description=current_desc,
                                )
                                self.proteins[current_id] = protein
                                count += 1

                        # Parse header
                        header = line[1:]
                        parts = header.split("|")
                        if len(parts) >= 2:
                            current_id = parts[1]
                            current_desc = parts[2] if len(parts) > 2 else None
                        else:
                            parts = header.split()
                            current_id = parts[0]
                            current_desc = (
                                " ".join(parts[1:]) if len(parts) > 1 else None
                            )

                        current_seq = []
                    else:
                        current_seq.append(line)

                # Save last protein
                if current_id and current_seq:
                    seq = "".join(current_seq)
                    if self._is_valid_protein(seq):
                        protein = ProteinInfo(
                            uniprot_id=current_id,
                            sequence=seq,
                            description=current_desc,
                        )
                        self.proteins[current_id] = protein
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} sequences from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_json_proteins(self, filepath: Path) -> int:
        """Load proteins from JSON file"""
        count = 0

        try:
            with open(filepath) as f:
                data = json.load(f)

            proteins_list = (
                data
                if isinstance(data, list)
                else data.get("proteins", data.get("targets", []))
            )

            for entry in proteins_list:
                if isinstance(entry, dict):
                    protein_id = (
                        entry.get("uniprot_id")
                        or entry.get("id")
                        or entry.get("accession")
                        or entry.get("target_id")
                    )
                    sequence = entry.get("sequence")

                    if protein_id and sequence and self._is_valid_protein(sequence):
                        protein = ProteinInfo(
                            uniprot_id=protein_id,
                            sequence=sequence,
                            gene_name=entry.get("gene_name"),
                            description=entry.get("description")
                            or entry.get("pref_name"),
                            target_type=entry.get("target_type"),
                        )
                        self.proteins[protein_id] = protein
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} proteins from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _is_valid_protein(self, sequence: str) -> bool:
        """Check if protein sequence is valid"""
        if not sequence:
            return False

        if len(sequence) < self.min_protein_length:
            return False

        if len(sequence) > self.max_protein_length:
            return False

        sequence_upper = sequence.upper()
        invalid_chars = set(sequence_upper) - self.EXTENDED_AA
        if invalid_chars:
            return False

        return True

    def load_molecules(self) -> int:
        """Load molecule data from various sources"""
        logger.info("Loading molecules...")

        total_loaded = 0

        # Load from molecule JSON files
        for mol_file in self.data_dir.glob("**/*molecules*.json"):
            count = self._load_json_molecules(mol_file)
            total_loaded += count

        # Load from compound files
        for compound_file in self.data_dir.glob("**/*compounds*.json"):
            count = self._load_json_molecules(compound_file)
            total_loaded += count

        # Load from SMILES files
        for smiles_file in self.data_dir.glob("**/*smiles*.txt"):
            count = self._load_smiles_file(smiles_file)
            total_loaded += count

        # Load from TSV files
        for tsv_file in self.data_dir.glob("**/*molecules*.tsv"):
            count = self._load_molecules_tsv(tsv_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.molecules):,} unique molecules")
        return len(self.molecules)

    def _load_json_molecules(self, filepath: Path) -> int:
        """Load molecules from JSON file"""
        count = 0

        try:
            with open(filepath) as f:
                data = json.load(f)

            mol_list = (
                data
                if isinstance(data, list)
                else data.get("molecules", data.get("compounds", []))
            )

            for entry in mol_list:
                if isinstance(entry, dict):
                    mol_id = (
                        entry.get("molecule_id")
                        or entry.get("chembl_id")
                        or entry.get("id")
                        or entry.get("compound_id")
                    )
                    smiles = (
                        entry.get("smiles")
                        or entry.get("canonical_smiles")
                        or entry.get("structure")
                    )

                    if mol_id and smiles and self._is_valid_smiles(smiles):
                        molecule = MoleculeInfo(
                            molecule_id=mol_id,
                            smiles=smiles,
                            name=entry.get("name") or entry.get("pref_name"),
                            molecular_weight=entry.get("molecular_weight")
                            or entry.get("mw"),
                            source=entry.get("source", "JSON"),
                        )
                        self.molecules[mol_id] = molecule
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} molecules from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_smiles_file(self, filepath: Path) -> int:
        """Load molecules from SMILES text file"""
        count = 0

        try:
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        smiles = parts[0]
                        mol_id = parts[1] if len(parts) > 1 else f"MOL_{count}"

                        if self._is_valid_smiles(smiles):
                            molecule = MoleculeInfo(
                                molecule_id=mol_id,
                                smiles=smiles,
                                source="SMILES_file",
                            )
                            self.molecules[mol_id] = molecule
                            count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} molecules from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_molecules_tsv(self, filepath: Path) -> int:
        """Load molecules from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline().lower()

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        mol_id = parts[0]
                        smiles = parts[1]

                        if self._is_valid_smiles(smiles):
                            molecule = MoleculeInfo(
                                molecule_id=mol_id,
                                smiles=smiles,
                                source="TSV",
                            )
                            self.molecules[mol_id] = molecule
                            count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} molecules from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid"""
        if not smiles:
            return False

        if len(smiles) < self.min_smiles_length:
            return False

        if len(smiles) > self.max_smiles_length:
            return False

        # Basic SMILES validation
        # Check for balanced parentheses and brackets
        if smiles.count("(") != smiles.count(")"):
            return False
        if smiles.count("[") != smiles.count("]"):
            return False

        # Check for valid characters (basic check)
        # More sophisticated validation would use RDKit
        invalid_chars = set(smiles) - SMILES_CHARS
        if invalid_chars:
            # Allow some additional characters that might appear
            extra_allowed = set("XxNnOoSsPpFfClBrIi")
            if invalid_chars - extra_allowed:
                return False

        return True

    def load_interactions(self) -> int:
        """Load protein-molecule interactions from various sources"""
        logger.info("Loading protein-molecule interactions...")

        total_loaded = 0

        # Load from interaction/activity files
        for interaction_file in self.data_dir.glob("**/*interactions*.tsv"):
            count = self._load_interaction_tsv(interaction_file)
            total_loaded += count

        # Load from activity files
        for activity_file in self.data_dir.glob("**/*activities*.json"):
            count = self._load_activities_json(activity_file)
            total_loaded += count

        # Load from binding files
        for binding_file in self.data_dir.glob("**/*binding*.tsv"):
            count = self._load_interaction_tsv(binding_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.interactions):,} positive interactions")
        return len(self.interactions)

    def _load_interaction_tsv(self, filepath: Path) -> int:
        """Load interactions from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline().lower()

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        protein_id = parts[0]
                        protein_seq = parts[1]
                        smiles = parts[2]
                        label = float(parts[3])

                        # Get affinity if available
                        affinity = float(parts[4]) if len(parts) > 4 else 0.0

                        if label >= 0.5:  # Positive interaction
                            # Validate sequences
                            if not self._is_valid_protein(protein_seq):
                                continue
                            if not self._is_valid_smiles(smiles):
                                continue

                            # Add protein if not exists
                            if protein_id not in self.proteins:
                                self.proteins[protein_id] = ProteinInfo(
                                    uniprot_id=protein_id,
                                    sequence=protein_seq,
                                )

                            # Create interaction
                            pair_key = (protein_seq, smiles)
                            if pair_key not in self.protein_mol_pairs:
                                source = (
                                    parts[5] if len(parts) > 5 else "interaction_file"
                                )
                                interaction = InteractionPair(
                                    protein_id=protein_id,
                                    protein_seq=protein_seq,
                                    smiles=smiles,
                                    label=1,
                                    affinity=affinity,
                                    source=source,
                                )
                                self.interactions.append(interaction)
                                self.protein_mol_pairs.add(pair_key)
                                count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} interactions from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_activities_json(self, filepath: Path) -> int:
        """Load activities from JSON file"""
        count = 0

        try:
            with open(filepath) as f:
                data = json.load(f)

            activities = data if isinstance(data, list) else data.get("activities", [])

            for entry in activities:
                if isinstance(entry, dict):
                    # Get protein info
                    protein_id = entry.get("target_id") or entry.get("uniprot_id")
                    protein_seq = entry.get("sequence")

                    # Get molecule info
                    smiles = entry.get("smiles") or entry.get("canonical_smiles")

                    # Get activity value
                    activity_value = entry.get("value") or entry.get("pchembl_value")
                    activity_type = entry.get("type") or entry.get("activity_type")

                    if protein_id and protein_seq and smiles:
                        # Check if activity indicates binding
                        is_active = False
                        affinity = 0.0

                        if activity_value:
                            try:
                                affinity = float(activity_value)
                                is_active = affinity >= self.affinity_threshold
                            except ValueError:
                                continue

                        if is_active:
                            if not self._is_valid_protein(protein_seq):
                                continue
                            if not self._is_valid_smiles(smiles):
                                continue

                            # Add protein if not exists
                            if protein_id not in self.proteins:
                                self.proteins[protein_id] = ProteinInfo(
                                    uniprot_id=protein_id,
                                    sequence=protein_seq,
                                )

                            pair_key = (protein_seq, smiles)
                            if pair_key not in self.protein_mol_pairs:
                                interaction = InteractionPair(
                                    protein_id=protein_id,
                                    protein_seq=protein_seq,
                                    smiles=smiles,
                                    label=1,
                                    affinity=affinity,
                                    source="ChEMBL",
                                    activity_type=activity_type,
                                )
                                self.interactions.append(interaction)
                                self.protein_mol_pairs.add(pair_key)
                                count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} activities from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def generate_negative_samples(self) -> int:
        """Generate negative interaction samples"""
        logger.info("Generating negative samples...")

        # Collect unique proteins and molecules from positive interactions
        unique_proteins = {}
        unique_smiles = set()

        for interaction in self.interactions:
            unique_proteins[interaction.protein_seq] = interaction.protein_id
            unique_smiles.add(interaction.smiles)

        protein_list = list(unique_proteins.keys())
        smiles_list = list(unique_smiles)

        if not protein_list or not smiles_list:
            logger.warning("No proteins or molecules available for negative sampling")
            return 0

        # Calculate number of negatives
        num_negatives = int(len(self.interactions) * self.negative_ratio)

        logger.info(f"  Generating {num_negatives:,} negative samples...")

        negative_interactions = []
        attempts = 0
        max_attempts = num_negatives * 20

        while len(negative_interactions) < num_negatives and attempts < max_attempts:
            protein_seq = random.choice(protein_list)
            smiles = random.choice(smiles_list)

            pair_key = (protein_seq, smiles)

            if pair_key not in self.protein_mol_pairs:
                protein_id = unique_proteins[protein_seq]
                interaction = InteractionPair(
                    protein_id=protein_id,
                    protein_seq=protein_seq,
                    smiles=smiles,
                    label=0,
                    affinity=0.0,
                    source="negative_sampling",
                )
                negative_interactions.append(interaction)
                self.protein_mol_pairs.add(pair_key)

            attempts += 1

        self.interactions.extend(negative_interactions)

        logger.info(f"  ✓ Generated {len(negative_interactions):,} negative samples")
        return len(negative_interactions)

    def compute_features(self) -> Dict[str, Any]:
        """Compute sequence-level features and statistics"""
        logger.info("Computing features...")

        features = {}

        # Protein length statistics
        protein_lengths = [len(i.protein_seq) for i in self.interactions]
        features["protein_length_stats"] = {
            "min": int(min(protein_lengths)),
            "max": int(max(protein_lengths)),
            "mean": float(np.mean(protein_lengths)),
            "std": float(np.std(protein_lengths)),
            "median": float(np.median(protein_lengths)),
        }

        # SMILES length statistics
        smiles_lengths = [len(i.smiles) for i in self.interactions]
        features["smiles_length_stats"] = {
            "min": int(min(smiles_lengths)),
            "max": int(max(smiles_lengths)),
            "mean": float(np.mean(smiles_lengths)),
            "std": float(np.std(smiles_lengths)),
            "median": float(np.median(smiles_lengths)),
        }

        # Interaction statistics
        positive_count = sum(1 for i in self.interactions if i.label == 1)
        negative_count = sum(1 for i in self.interactions if i.label == 0)

        features["interaction_stats"] = {
            "total_interactions": len(self.interactions),
            "positive_interactions": positive_count,
            "negative_interactions": negative_count,
            "positive_ratio": positive_count / len(self.interactions)
            if self.interactions
            else 0,
        }

        # Unique sequences
        unique_proteins = set(i.protein_seq for i in self.interactions)
        unique_smiles = set(i.smiles for i in self.interactions)

        features["unique_stats"] = {
            "unique_proteins": len(unique_proteins),
            "unique_molecules": len(unique_smiles),
        }

        # Affinity statistics for positive interactions
        positive_affinities = [
            i.affinity for i in self.interactions if i.label == 1 and i.affinity > 0
        ]
        if positive_affinities:
            features["affinity_stats"] = {
                "min": float(min(positive_affinities)),
                "max": float(max(positive_affinities)),
                "mean": float(np.mean(positive_affinities)),
                "std": float(np.std(positive_affinities)),
            }

        logger.info(
            f"  ✓ Computed features for {len(self.interactions):,} interactions"
        )
        return features

    def split_data(self) -> DataSplit:
        """Split data into train/val/test sets"""
        logger.info("Splitting data...")

        # Shuffle interactions
        shuffled = self.interactions.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_pairs = shuffled[:train_end]
        val_pairs = shuffled[train_end:val_end]
        test_pairs = shuffled[val_end:]

        split = DataSplit(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
        )

        # Log split statistics
        train_pos = sum(1 for p in train_pairs if p.label == 1)
        val_pos = sum(1 for p in val_pairs if p.label == 1)
        test_pos = sum(1 for p in test_pairs if p.label == 1)

        logger.info(f"  ✓ Train: {len(train_pairs):,} pairs ({train_pos:,} positive)")
        logger.info(f"  ✓ Val: {len(val_pairs):,} pairs ({val_pos:,} positive)")
        logger.info(f"  ✓ Test: {len(test_pairs):,} pairs ({test_pos:,} positive)")

        return split

    def save_prepared_data(
        self,
        split: DataSplit,
        features: Dict[str, Any],
    ) -> None:
        """Save prepared data to output directory"""
        logger.info("Saving prepared data...")

        # Save interaction pairs as TSV files
        for name, pairs in [
            ("train", split.train_pairs),
            ("val", split.val_pairs),
            ("test", split.test_pairs),
        ]:
            filepath = self.output_dir / f"p2m_{name}.tsv"
            self._save_pairs_tsv(filepath, pairs)

        # Save all interactions
        all_filepath = self.output_dir / "p2m_all.tsv"
        self._save_pairs_tsv(all_filepath, self.interactions)

        # Save unique sequences
        unique_proteins = {}
        unique_molecules = {}

        for interaction in self.interactions:
            if interaction.protein_id not in unique_proteins:
                unique_proteins[interaction.protein_id] = interaction.protein_seq
            mol_id = f"MOL_{len(unique_molecules)}"
            if interaction.smiles not in unique_molecules.values():
                unique_molecules[mol_id] = interaction.smiles

        # Save protein sequences
        proteins_json = {
            pid: {"sequence": seq, "length": len(seq)}
            for pid, seq in unique_proteins.items()
        }
        with open(self.output_dir / "p2m_proteins.json", "w") as f:
            json.dump(proteins_json, f, indent=2)

        # Save molecule SMILES
        molecules_json = {
            mid: {"smiles": smiles, "length": len(smiles)}
            for mid, smiles in unique_molecules.items()
        }
        with open(self.output_dir / "p2m_molecules.json", "w") as f:
            json.dump(molecules_json, f, indent=2)

        # Save features and metadata
        metadata = {
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "min_protein_length": self.min_protein_length,
            "max_protein_length": self.max_protein_length,
            "min_smiles_length": self.min_smiles_length,
            "max_smiles_length": self.max_smiles_length,
            "affinity_threshold": self.affinity_threshold,
            "negative_ratio": self.negative_ratio,
            "split_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "split_sizes": {
                "train": len(split.train_pairs),
                "val": len(split.val_pairs),
                "test": len(split.test_pairs),
            },
            "features": features,
        }

        metadata_filepath = self.output_dir / "p2m_metadata.json"
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save as pickle for faster loading
        pickle_data = {
            "train": [
                (p.protein_id, p.protein_seq, p.smiles, p.label, p.affinity)
                for p in split.train_pairs
            ],
            "val": [
                (p.protein_id, p.protein_seq, p.smiles, p.label, p.affinity)
                for p in split.val_pairs
            ],
            "test": [
                (p.protein_id, p.protein_seq, p.smiles, p.label, p.affinity)
                for p in split.test_pairs
            ],
            "proteins": proteins_json,
            "molecules": molecules_json,
            "metadata": metadata,
        }

        pickle_filepath = self.output_dir / "p2m_prepared.pkl"
        with open(pickle_filepath, "wb") as f:
            pickle.dump(pickle_data, f)

        logger.info(f"  ✓ Saved TSV files: p2m_{{train,val,test,all}}.tsv")
        logger.info(f"  ✓ Saved sequences: p2m_proteins.json, p2m_molecules.json")
        logger.info(f"  ✓ Saved metadata: p2m_metadata.json")
        logger.info(f"  ✓ Saved pickle: p2m_prepared.pkl")

    def _save_pairs_tsv(self, filepath: Path, pairs: List[InteractionPair]) -> None:
        """Save interaction pairs to TSV file"""
        with open(filepath, "w") as f:
            f.write("protein_id\tprotein_seq\tsmiles\tlabel\taffinity\tsource\n")

            for pair in pairs:
                f.write(
                    f"{pair.protein_id}\t{pair.protein_seq}\t{pair.smiles}\t"
                    f"{pair.label}\t{pair.affinity:.4f}\t{pair.source}\n"
                )

    def prepare(self) -> Dict[str, Any]:
        """Run full data preparation pipeline"""
        logger.info("=" * 60)
        logger.info("Preparing P2M (Protein-Molecule Interaction) Data")
        logger.info("=" * 60)

        # Load data
        self.load_proteins()
        self.load_molecules()
        self.load_interactions()

        # Check if we have enough data
        if len(self.interactions) == 0:
            logger.warning(
                "No interactions found! Creating synthetic data for testing..."
            )
            self._create_synthetic_data()

        # Generate negative samples
        self.generate_negative_samples()

        # Compute features
        features = self.compute_features()

        # Split data
        split = self.split_data()

        # Save prepared data
        self.save_prepared_data(split, features)

        # Print summary
        self._print_summary(features)

        return {
            "num_proteins": len(self.proteins),
            "num_molecules": len(self.molecules),
            "num_interactions": len(self.interactions),
            "train_size": len(split.train_pairs),
            "val_size": len(split.val_pairs),
            "test_size": len(split.test_pairs),
            "output_dir": str(self.output_dir),
        }

    def _create_synthetic_data(self) -> None:
        """Create synthetic data for testing when no real data is available"""
        logger.info("Creating synthetic protein-molecule data for testing...")

        # Generate synthetic proteins (drug targets)
        for i in range(50):
            protein_id = f"SYNTH_TARGET_{i:04d}"
            length = random.randint(100, 500)
            sequence = "".join(random.choices(list(self.AMINO_ACIDS), k=length))

            self.proteins[protein_id] = ProteinInfo(
                uniprot_id=protein_id,
                sequence=sequence,
                gene_name=f"TARGET_{i}",
            )

        # Generate synthetic molecules (drug-like SMILES)
        synthetic_smiles_templates = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen-like
            "c1ccc2[nH]ccc2c1",  # Indole
            "c1ccc2ccccc2c1",  # Naphthalene
        ]

        for i in range(100):
            mol_id = f"SYNTH_MOL_{i:04d}"
            base_smiles = random.choice(synthetic_smiles_templates)
            # Add some variation
            smiles = base_smiles

            self.molecules[mol_id] = MoleculeInfo(
                molecule_id=mol_id,
                smiles=smiles,
                source="synthetic",
            )

        # Generate synthetic interactions
        protein_list = list(self.proteins.keys())
        mol_list = list(self.molecules.keys())

        for _ in range(200):
            protein_id = random.choice(protein_list)
            mol_id = random.choice(mol_list)

            protein_seq = self.proteins[protein_id].sequence
            smiles = self.molecules[mol_id].smiles

            pair_key = (protein_seq, smiles)
            if pair_key not in self.protein_mol_pairs:
                affinity = random.uniform(5.0, 10.0)  # Simulated pIC50
                self.interactions.append(
                    InteractionPair(
                        protein_id=protein_id,
                        protein_seq=protein_seq,
                        smiles=smiles,
                        label=1,
                        affinity=affinity,
                        source="synthetic",
                    )
                )
                self.protein_mol_pairs.add(pair_key)

        logger.info(f"  ✓ Created {len(self.proteins)} synthetic proteins")
        logger.info(f"  ✓ Created {len(self.molecules)} synthetic molecules")
        logger.info(f"  ✓ Created {len(self.interactions)} synthetic interactions")

    def _print_summary(self, features: Dict[str, Any]) -> None:
        """Print summary of prepared data"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Data Preparation Summary")
        logger.info("=" * 60)

        stats = features.get("interaction_stats", {})
        protein_stats = features.get("protein_length_stats", {})
        smiles_stats = features.get("smiles_length_stats", {})
        unique_stats = features.get("unique_stats", {})
        affinity_stats = features.get("affinity_stats", {})

        logger.info(f"Total interactions: {stats.get('total_interactions', 0):,}")
        logger.info(f"  - Positive: {stats.get('positive_interactions', 0):,}")
        logger.info(f"  - Negative: {stats.get('negative_interactions', 0):,}")
        logger.info(f"Unique proteins: {unique_stats.get('unique_proteins', 0):,}")
        logger.info(f"Unique molecules: {unique_stats.get('unique_molecules', 0):,}")
        logger.info(
            f"Protein length range: {protein_stats.get('min', 0)}-{protein_stats.get('max', 0)}"
        )
        logger.info(
            f"SMILES length range: {smiles_stats.get('min', 0)}-{smiles_stats.get('max', 0)}"
        )
        if affinity_stats:
            logger.info(
                f"Affinity range: {affinity_stats.get('min', 0):.2f}-{affinity_stats.get('max', 0):.2f}"
            )
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Protein-Molecule Interaction data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing raw downloaded data",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save prepared data",
    )

    parser.add_argument(
        "--min-protein-length",
        type=int,
        default=20,
        help="Minimum protein sequence length",
    )

    parser.add_argument(
        "--max-protein-length",
        type=int,
        default=2000,
        help="Maximum protein sequence length",
    )

    parser.add_argument(
        "--min-smiles-length",
        type=int,
        default=3,
        help="Minimum SMILES string length",
    )

    parser.add_argument(
        "--max-smiles-length",
        type=int,
        default=200,
        help="Maximum SMILES string length",
    )

    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=5.0,
        help="Minimum affinity (pIC50/pKd) for positive interactions",
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for testing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.warning(f"Split ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # Create preparer and run
    preparer = PMolDataPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_protein_length=args.min_protein_length,
        max_protein_length=args.max_protein_length,
        min_smiles_length=args.min_smiles_length,
        max_smiles_length=args.max_smiles_length,
        affinity_threshold=args.affinity_threshold,
        negative_ratio=args.negative_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    result = preparer.prepare()

    logger.info("")
    logger.info("Data preparation complete!")
    logger.info(f"Output saved to: {result['output_dir']}")

    return result


if __name__ == "__main__":
    main()
