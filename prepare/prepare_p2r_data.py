#!/usr/bin/env python3
"""
Prepare Protein-RNA Interaction (P2R) Data for Training

This script processes raw downloaded protein-RNA interaction data and prepares it for
training the P2R interaction prediction model.

Steps:
1. Load protein sequences from UniProt downloads
2. Load RNA sequences from ATtRACT and other databases
3. Load protein-RNA interaction pairs
4. Filter and validate sequences
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared" / "p2r"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# RNA nucleotide mappings
RNA_NUCLEOTIDES = set("ACGU")
IUPAC_RNA = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "U": ["U"],
    "R": ["A", "G"],
    "Y": ["C", "U"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "K": ["G", "U"],
    "M": ["A", "C"],
    "B": ["C", "G", "U"],
    "D": ["A", "G", "U"],
    "H": ["A", "C", "U"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "U"],
}


@dataclass
class ProteinInfo:
    """Store protein information"""

    uniprot_id: str
    sequence: str
    gene_name: Optional[str] = None
    description: Optional[str] = None
    rna_binding_domain: Optional[str] = None
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class RNAMotif:
    """Store RNA motif information"""

    motif_id: str
    sequence: str
    protein_name: Optional[str] = None
    motif_type: Optional[str] = None  # e.g., "stem-loop", "hairpin", etc.
    source: str = "ATtRACT"
    score: float = 1.0

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class InteractionPair:
    """Store protein-RNA interaction pair information"""

    protein_id: str
    protein_seq: str
    rna_seq: str
    label: int  # 1 for positive, 0 for negative
    source: str = "ATtRACT"
    score: float = 1.0
    evidence_type: Optional[str] = None


@dataclass
class DataSplit:
    """Store data split information"""

    train_pairs: List[InteractionPair]
    val_pairs: List[InteractionPair]
    test_pairs: List[InteractionPair]

    @property
    def total_pairs(self) -> int:
        return len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)


class PRNADataPreparer:
    """Prepare protein-RNA interaction data for training"""

    # Standard amino acids
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        min_protein_length: int = 20,
        max_protein_length: int = 2000,
        min_rna_length: int = 6,
        max_rna_length: int = 500,
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
        self.min_rna_length = min_rna_length
        self.max_rna_length = max_rna_length
        self.negative_ratio = negative_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Data containers
        self.proteins: Dict[str, ProteinInfo] = {}
        self.rna_motifs: Dict[str, RNAMotif] = {}
        self.interactions: List[InteractionPair] = []
        self.protein_rna_pairs: Set[Tuple[str, str]] = set()

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
            "**/*rna_binding*.fasta",
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

            proteins_list = data if isinstance(data, list) else data.get("proteins", [])

            for entry in proteins_list:
                if isinstance(entry, dict):
                    protein_id = (
                        entry.get("uniprot_id")
                        or entry.get("id")
                        or entry.get("accession")
                    )
                    sequence = entry.get("sequence")

                    if protein_id and sequence and self._is_valid_protein(sequence):
                        protein = ProteinInfo(
                            uniprot_id=protein_id,
                            sequence=sequence,
                            gene_name=entry.get("gene_name"),
                            description=entry.get("description"),
                            rna_binding_domain=entry.get("rna_binding_domain"),
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

    def load_rna_motifs(self) -> int:
        """Load RNA motifs from ATtRACT and other sources"""
        logger.info("Loading RNA motifs...")

        total_loaded = 0

        # Load from ATtRACT database files
        for attract_file in self.data_dir.glob("**/*attract*.tsv"):
            count = self._load_attract_motifs(attract_file)
            total_loaded += count

        # Load from RNA motif JSON files
        for motif_file in self.data_dir.glob("**/*rna*motif*.json"):
            count = self._load_json_rna_motifs(motif_file)
            total_loaded += count

        # Load from FASTA files containing RNA
        for fasta_file in self.data_dir.glob("**/*rna*.fasta"):
            count = self._load_rna_fasta(fasta_file)
            total_loaded += count

        # Load from sequences TSV
        for seq_file in self.data_dir.glob("**/*sequences*.tsv"):
            count = self._load_rna_sequences_tsv(seq_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.rna_motifs):,} unique RNA motifs")
        return len(self.rna_motifs)

    def _load_attract_motifs(self, filepath: Path) -> int:
        """Load motifs from ATtRACT database file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline()

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        motif_id = parts[0]
                        sequence = parts[1] if len(parts) > 1 else None
                        protein_name = parts[2] if len(parts) > 2 else None

                        if sequence:
                            # Convert T to U for RNA
                            sequence = sequence.upper().replace("T", "U")

                            if self._is_valid_rna(sequence):
                                motif = RNAMotif(
                                    motif_id=motif_id,
                                    sequence=sequence,
                                    protein_name=protein_name,
                                    source="ATtRACT",
                                )
                                self.rna_motifs[motif_id] = motif
                                count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} motifs from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_json_rna_motifs(self, filepath: Path) -> int:
        """Load RNA motifs from JSON file"""
        count = 0

        try:
            with open(filepath) as f:
                data = json.load(f)

            motifs_list = data if isinstance(data, list) else data.get("motifs", [])

            for entry in motifs_list:
                if isinstance(entry, dict):
                    motif_id = entry.get("id") or entry.get("motif_id")
                    sequence = entry.get("sequence")

                    if motif_id and sequence:
                        sequence = sequence.upper().replace("T", "U")

                        if self._is_valid_rna(sequence):
                            motif = RNAMotif(
                                motif_id=motif_id,
                                sequence=sequence,
                                protein_name=entry.get("protein_name"),
                                motif_type=entry.get("motif_type"),
                                source=entry.get("source", "JSON"),
                            )
                            self.rna_motifs[motif_id] = motif
                            count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} motifs from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_rna_fasta(self, filepath: Path) -> int:
        """Load RNA sequences from FASTA file"""
        count = 0

        try:
            open_func = gzip.open if filepath.suffix == ".gz" else open
            mode = "rt" if filepath.suffix == ".gz" else "r"

            with open_func(filepath, mode) as f:
                current_id = None
                current_seq = []

                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_id and current_seq:
                            seq = "".join(current_seq).upper().replace("T", "U")
                            if self._is_valid_rna(seq):
                                motif = RNAMotif(
                                    motif_id=current_id,
                                    sequence=seq,
                                    source="FASTA",
                                )
                                self.rna_motifs[current_id] = motif
                                count += 1

                        parts = line[1:].split()
                        current_id = parts[0]
                        current_seq = []
                    else:
                        current_seq.append(line)

                # Save last sequence
                if current_id and current_seq:
                    seq = "".join(current_seq).upper().replace("T", "U")
                    if self._is_valid_rna(seq):
                        motif = RNAMotif(
                            motif_id=current_id,
                            sequence=seq,
                            source="FASTA",
                        )
                        self.rna_motifs[current_id] = motif
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} RNA sequences from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_rna_sequences_tsv(self, filepath: Path) -> int:
        """Load RNA sequences from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline().lower()

                # Check if this file contains RNA sequences
                if "rna" not in header and "sequence" not in header:
                    return 0

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        seq_id = parts[0]
                        sequence = parts[1]

                        if sequence:
                            sequence = sequence.upper().replace("T", "U")

                            if self._is_valid_rna(sequence):
                                motif = RNAMotif(
                                    motif_id=seq_id,
                                    sequence=sequence,
                                    source="TSV",
                                )
                                self.rna_motifs[seq_id] = motif
                                count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} RNA sequences from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _expand_iupac(self, sequence: str) -> List[str]:
        """Expand IUPAC ambiguity codes to concrete sequences"""
        if not sequence:
            return []

        sequences = [""]
        for char in sequence.upper():
            if char in IUPAC_RNA:
                new_sequences = []
                for seq in sequences:
                    for nucleotide in IUPAC_RNA[char]:
                        new_sequences.append(seq + nucleotide)
                sequences = new_sequences
            else:
                sequences = [seq + char for seq in sequences]

            # Limit expansion to prevent combinatorial explosion
            if len(sequences) > 100:
                sequences = sequences[:100]
                break

        return sequences

    def _is_valid_rna(self, sequence: str) -> bool:
        """Check if RNA sequence is valid"""
        if not sequence:
            return False

        if len(sequence) < self.min_rna_length:
            return False

        if len(sequence) > self.max_rna_length:
            return False

        sequence_upper = sequence.upper()
        invalid_chars = set(sequence_upper) - RNA_NUCLEOTIDES
        if invalid_chars:
            return False

        return True

    def load_interactions(self) -> int:
        """Load protein-RNA interactions from various sources"""
        logger.info("Loading protein-RNA interactions...")

        total_loaded = 0

        # Load from interaction TSV files
        for interaction_file in self.data_dir.glob("**/*interactions*.tsv"):
            count = self._load_interaction_tsv(interaction_file)
            total_loaded += count

        # Load from protein-RNA pair files
        for pair_file in self.data_dir.glob("**/*protein_rna*.tsv"):
            count = self._load_interaction_tsv(pair_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.interactions):,} positive interactions")
        return len(self.interactions)

    def _load_interaction_tsv(self, filepath: Path) -> int:
        """Load interactions from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline().lower()

                # Check if this file might contain RNA interactions
                if "rna" not in filepath.name.lower() and "rna" not in header:
                    return 0

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        protein_id = parts[0]
                        protein_seq = parts[1]
                        rna_seq = parts[2].upper().replace("T", "U")
                        label = int(parts[3])

                        if label == 1:
                            # Validate sequences
                            if not self._is_valid_protein(protein_seq):
                                continue
                            if not self._is_valid_rna(rna_seq):
                                continue

                            # Add protein if not exists
                            if protein_id not in self.proteins:
                                self.proteins[protein_id] = ProteinInfo(
                                    uniprot_id=protein_id,
                                    sequence=protein_seq,
                                )

                            # Create interaction
                            pair_key = (protein_seq, rna_seq)
                            if pair_key not in self.protein_rna_pairs:
                                source = (
                                    parts[4] if len(parts) > 4 else "interaction_file"
                                )
                                interaction = InteractionPair(
                                    protein_id=protein_id,
                                    protein_seq=protein_seq,
                                    rna_seq=rna_seq,
                                    label=1,
                                    source=source,
                                )
                                self.interactions.append(interaction)
                                self.protein_rna_pairs.add(pair_key)
                                count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} interactions from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def generate_negative_samples(self) -> int:
        """Generate negative interaction samples"""
        logger.info("Generating negative samples...")

        # Collect unique proteins and RNA sequences from positive interactions
        unique_proteins = {}
        unique_rna = set()

        for interaction in self.interactions:
            unique_proteins[interaction.protein_seq] = interaction.protein_id
            unique_rna.add(interaction.rna_seq)

        protein_list = list(unique_proteins.keys())
        rna_list = list(unique_rna)

        if not protein_list or not rna_list:
            logger.warning(
                "No proteins or RNA sequences available for negative sampling"
            )
            return 0

        # Calculate number of negatives
        num_negatives = int(len(self.interactions) * self.negative_ratio)

        logger.info(f"  Generating {num_negatives:,} negative samples...")

        negative_interactions = []
        attempts = 0
        max_attempts = num_negatives * 20

        while len(negative_interactions) < num_negatives and attempts < max_attempts:
            protein_seq = random.choice(protein_list)
            rna_seq = random.choice(rna_list)

            pair_key = (protein_seq, rna_seq)

            if pair_key not in self.protein_rna_pairs:
                protein_id = unique_proteins[protein_seq]
                interaction = InteractionPair(
                    protein_id=protein_id,
                    protein_seq=protein_seq,
                    rna_seq=rna_seq,
                    label=0,
                    source="negative_sampling",
                )
                negative_interactions.append(interaction)
                self.protein_rna_pairs.add(pair_key)

            attempts += 1

        self.interactions.extend(negative_interactions)

        logger.info(f"  ✓ Generated {len(negative_interactions):,} negative samples")
        return len(negative_interactions)

    def augment_data(self) -> int:
        """Augment data with sequence variations"""
        logger.info("Augmenting data...")

        augmented_count = 0

        # RNA sequence augmentation strategies
        positive_interactions = [i for i in self.interactions if i.label == 1]

        for interaction in positive_interactions[:1000]:  # Limit augmentation
            # 1. Add slight mutations (1-2 positions)
            mutated_rna = self._mutate_rna(interaction.rna_seq, num_mutations=1)

            if mutated_rna and mutated_rna != interaction.rna_seq:
                pair_key = (interaction.protein_seq, mutated_rna)
                if pair_key not in self.protein_rna_pairs:
                    aug_interaction = InteractionPair(
                        protein_id=interaction.protein_id,
                        protein_seq=interaction.protein_seq,
                        rna_seq=mutated_rna,
                        label=1,
                        source="augmented_mutation",
                        score=0.9,  # Slightly lower confidence
                    )
                    self.interactions.append(aug_interaction)
                    self.protein_rna_pairs.add(pair_key)
                    augmented_count += 1

        logger.info(f"  ✓ Added {augmented_count:,} augmented samples")
        return augmented_count

    def _mutate_rna(self, sequence: str, num_mutations: int = 1) -> str:
        """Apply random mutations to RNA sequence"""
        if len(sequence) < 3:
            return ""

        nucleotides = list("ACGU")
        seq_list = list(sequence)

        for _ in range(num_mutations):
            pos = random.randint(0, len(seq_list) - 1)
            current = seq_list[pos]
            alternatives = [n for n in nucleotides if n != current]
            seq_list[pos] = random.choice(alternatives)

        return "".join(seq_list)

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

        # RNA length statistics
        rna_lengths = [len(i.rna_seq) for i in self.interactions]
        features["rna_length_stats"] = {
            "min": int(min(rna_lengths)),
            "max": int(max(rna_lengths)),
            "mean": float(np.mean(rna_lengths)),
            "std": float(np.std(rna_lengths)),
            "median": float(np.median(rna_lengths)),
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
        unique_rna = set(i.rna_seq for i in self.interactions)

        features["unique_stats"] = {
            "unique_proteins": len(unique_proteins),
            "unique_rna_sequences": len(unique_rna),
        }

        # RNA nucleotide composition
        all_rna = "".join(i.rna_seq for i in self.interactions)
        rna_counts = {nt: all_rna.count(nt) for nt in "ACGU"}
        total_nt = len(all_rna)

        features["rna_composition"] = {
            nt: count / total_nt for nt, count in rna_counts.items()
        }

        # GC content
        gc_content = (
            (rna_counts["G"] + rna_counts["C"]) / total_nt if total_nt > 0 else 0
        )
        features["gc_content"] = gc_content

        # AU content (important for RNA structure)
        au_content = (
            (rna_counts["A"] + rna_counts["U"]) / total_nt if total_nt > 0 else 0
        )
        features["au_content"] = au_content

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
            filepath = self.output_dir / f"p2r_{name}.tsv"
            self._save_pairs_tsv(filepath, pairs)

        # Save all interactions
        all_filepath = self.output_dir / "p2r_all.tsv"
        self._save_pairs_tsv(all_filepath, self.interactions)

        # Save unique sequences
        unique_proteins = {}
        unique_rna = {}

        for interaction in self.interactions:
            if interaction.protein_id not in unique_proteins:
                unique_proteins[interaction.protein_id] = interaction.protein_seq
            rna_id = f"RNA_{len(unique_rna)}"
            if interaction.rna_seq not in unique_rna.values():
                unique_rna[rna_id] = interaction.rna_seq

        # Save protein sequences
        proteins_json = {
            pid: {"sequence": seq, "length": len(seq)}
            for pid, seq in unique_proteins.items()
        }
        with open(self.output_dir / "p2r_proteins.json", "w") as f:
            json.dump(proteins_json, f, indent=2)

        # Save RNA sequences
        rna_json = {
            rid: {"sequence": seq, "length": len(seq)}
            for rid, seq in unique_rna.items()
        }
        with open(self.output_dir / "p2r_rna_sequences.json", "w") as f:
            json.dump(rna_json, f, indent=2)

        # Save features and metadata
        metadata = {
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "min_protein_length": self.min_protein_length,
            "max_protein_length": self.max_protein_length,
            "min_rna_length": self.min_rna_length,
            "max_rna_length": self.max_rna_length,
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

        metadata_filepath = self.output_dir / "p2r_metadata.json"
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save as pickle for faster loading
        pickle_data = {
            "train": [
                (p.protein_id, p.protein_seq, p.rna_seq, p.label, p.score)
                for p in split.train_pairs
            ],
            "val": [
                (p.protein_id, p.protein_seq, p.rna_seq, p.label, p.score)
                for p in split.val_pairs
            ],
            "test": [
                (p.protein_id, p.protein_seq, p.rna_seq, p.label, p.score)
                for p in split.test_pairs
            ],
            "proteins": proteins_json,
            "rna_sequences": rna_json,
            "metadata": metadata,
        }

        pickle_filepath = self.output_dir / "p2r_prepared.pkl"
        with open(pickle_filepath, "wb") as f:
            pickle.dump(pickle_data, f)

        logger.info(f"  ✓ Saved TSV files: p2r_{{train,val,test,all}}.tsv")
        logger.info(f"  ✓ Saved sequences: p2r_proteins.json, p2r_rna_sequences.json")
        logger.info(f"  ✓ Saved metadata: p2r_metadata.json")
        logger.info(f"  ✓ Saved pickle: p2r_prepared.pkl")

    def _save_pairs_tsv(self, filepath: Path, pairs: List[InteractionPair]) -> None:
        """Save interaction pairs to TSV file"""
        with open(filepath, "w") as f:
            f.write("protein_id\tprotein_seq\trna_seq\tlabel\tscore\tsource\n")

            for pair in pairs:
                f.write(
                    f"{pair.protein_id}\t{pair.protein_seq}\t{pair.rna_seq}\t"
                    f"{pair.label}\t{pair.score:.4f}\t{pair.source}\n"
                )

    def prepare(self) -> Dict[str, Any]:
        """Run full data preparation pipeline"""
        logger.info("=" * 60)
        logger.info("Preparing P2R (Protein-RNA Interaction) Data")
        logger.info("=" * 60)

        # Load data
        self.load_proteins()
        self.load_rna_motifs()
        self.load_interactions()

        # Check if we have enough data
        if len(self.interactions) == 0:
            logger.warning(
                "No interactions found! Creating synthetic data for testing..."
            )
            self._create_synthetic_data()

        # Generate negative samples
        self.generate_negative_samples()

        # Augment data
        self.augment_data()

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
            "num_rna_motifs": len(self.rna_motifs),
            "num_interactions": len(self.interactions),
            "train_size": len(split.train_pairs),
            "val_size": len(split.val_pairs),
            "test_size": len(split.test_pairs),
            "output_dir": str(self.output_dir),
        }

    def _create_synthetic_data(self) -> None:
        """Create synthetic data for testing when no real data is available"""
        logger.info("Creating synthetic protein-RNA data for testing...")

        # Generate synthetic proteins (RNA binding proteins)
        for i in range(50):
            protein_id = f"SYNTH_RBP_{i:04d}"
            length = random.randint(100, 500)
            sequence = "".join(random.choices(list(self.AMINO_ACIDS), k=length))

            self.proteins[protein_id] = ProteinInfo(
                uniprot_id=protein_id,
                sequence=sequence,
                gene_name=f"RBP_{i}",
            )

        # Generate synthetic RNA motifs
        for i in range(100):
            motif_id = f"SYNTH_RNA_{i:04d}"
            length = random.randint(8, 30)
            sequence = "".join(random.choices(list(RNA_NUCLEOTIDES), k=length))

            self.rna_motifs[motif_id] = RNAMotif(
                motif_id=motif_id,
                sequence=sequence,
                source="synthetic",
            )

        # Generate synthetic interactions
        protein_list = list(self.proteins.keys())
        motif_list = list(self.rna_motifs.keys())

        for _ in range(200):
            protein_id = random.choice(protein_list)
            motif_id = random.choice(motif_list)

            protein_seq = self.proteins[protein_id].sequence
            rna_seq = self.rna_motifs[motif_id].sequence

            pair_key = (protein_seq, rna_seq)
            if pair_key not in self.protein_rna_pairs:
                self.interactions.append(
                    InteractionPair(
                        protein_id=protein_id,
                        protein_seq=protein_seq,
                        rna_seq=rna_seq,
                        label=1,
                        source="synthetic",
                    )
                )
                self.protein_rna_pairs.add(pair_key)

        logger.info(f"  ✓ Created {len(self.proteins)} synthetic proteins")
        logger.info(f"  ✓ Created {len(self.rna_motifs)} synthetic RNA motifs")
        logger.info(f"  ✓ Created {len(self.interactions)} synthetic interactions")

    def _print_summary(self, features: Dict[str, Any]) -> None:
        """Print summary of prepared data"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Data Preparation Summary")
        logger.info("=" * 60)

        stats = features.get("interaction_stats", {})
        protein_stats = features.get("protein_length_stats", {})
        rna_stats = features.get("rna_length_stats", {})
        unique_stats = features.get("unique_stats", {})

        logger.info(f"Total interactions: {stats.get('total_interactions', 0):,}")
        logger.info(f"  - Positive: {stats.get('positive_interactions', 0):,}")
        logger.info(f"  - Negative: {stats.get('negative_interactions', 0):,}")
        logger.info(f"Unique proteins: {unique_stats.get('unique_proteins', 0):,}")
        logger.info(
            f"Unique RNA sequences: {unique_stats.get('unique_rna_sequences', 0):,}"
        )
        logger.info(
            f"Protein length range: {protein_stats.get('min', 0)}-{protein_stats.get('max', 0)}"
        )
        logger.info(
            f"RNA length range: {rna_stats.get('min', 0)}-{rna_stats.get('max', 0)}"
        )
        logger.info(f"GC content: {features.get('gc_content', 0):.2%}")
        logger.info(f"AU content: {features.get('au_content', 0):.2%}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Protein-RNA Interaction data for training",
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
        "--min-rna-length",
        type=int,
        default=6,
        help="Minimum RNA sequence length",
    )

    parser.add_argument(
        "--max-rna-length",
        type=int,
        default=500,
        help="Maximum RNA sequence length",
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
    preparer = PRNADataPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_protein_length=args.min_protein_length,
        max_protein_length=args.max_protein_length,
        min_rna_length=args.min_rna_length,
        max_rna_length=args.max_rna_length,
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
