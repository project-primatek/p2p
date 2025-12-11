#!/usr/bin/env python3
"""
Prepare Protein-DNA Interaction (P2D) Data for Training

This script processes raw downloaded protein-DNA interaction data and prepares it for
training the P2D interaction prediction model.

Steps:
1. Load protein sequences from UniProt downloads
2. Load DNA binding site data from JASPAR
3. Load protein-DNA interaction pairs
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared" / "p2d"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# DNA nucleotide mappings
DNA_NUCLEOTIDES = set("ACGT")
IUPAC_DNA = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}


@dataclass
class ProteinInfo:
    """Store protein information"""

    uniprot_id: str
    sequence: str
    gene_name: Optional[str] = None
    description: Optional[str] = None
    dna_binding_domain: Optional[str] = None
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class DNAMotif:
    """Store DNA motif information"""

    motif_id: str
    sequence: str
    tf_name: Optional[str] = None
    tf_class: Optional[str] = None
    source: str = "JASPAR"
    score: float = 1.0


@dataclass
class InteractionPair:
    """Store protein-DNA interaction pair information"""

    protein_id: str
    protein_seq: str
    dna_seq: str
    label: int  # 1 for positive, 0 for negative
    source: str = "JASPAR"
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


class PDNADataPreparer:
    """Prepare protein-DNA interaction data for training"""

    # Standard amino acids
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        min_protein_length: int = 20,
        max_protein_length: int = 2000,
        min_dna_length: int = 6,
        max_dna_length: int = 500,
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
        self.min_dna_length = min_dna_length
        self.max_dna_length = max_dna_length
        self.negative_ratio = negative_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Data containers
        self.proteins: Dict[str, ProteinInfo] = {}
        self.dna_motifs: Dict[str, DNAMotif] = {}
        self.interactions: List[InteractionPair] = []
        self.protein_dna_pairs: Set[Tuple[str, str]] = set()

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
            "**/*dna_binding*.fasta",
            "**/uniprot/*.fasta",
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
                            dna_binding_domain=entry.get("dna_binding_domain"),
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

    def load_dna_motifs(self) -> int:
        """Load DNA motifs from JASPAR and other sources"""
        logger.info("Loading DNA motifs...")

        total_loaded = 0

        # Load from JASPAR motif files
        for motif_file in self.data_dir.glob("**/jaspar/*motif*.json"):
            count = self._load_jaspar_motifs(motif_file)
            total_loaded += count

        # Load position frequency matrices
        for pfm_file in self.data_dir.glob("**/*frequency_matrices*.json"):
            count = self._load_pfm_motifs(pfm_file)
            total_loaded += count

        # Load binding sites
        for binding_file in self.data_dir.glob("**/*binding_sites*.tsv"):
            count = self._load_binding_sites(binding_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.dna_motifs):,} unique DNA motifs")
        return len(self.dna_motifs)

    def _load_jaspar_motifs(self, filepath: Path) -> int:
        """Load motifs from JASPAR JSON file"""
        count = 0

        try:
            with open(filepath) as f:
                data = json.load(f)

            motifs_list = data if isinstance(data, list) else data.get("motifs", [])

            for entry in motifs_list:
                if isinstance(entry, dict):
                    motif_id = entry.get("matrix_id") or entry.get("id")
                    consensus = entry.get("consensus") or entry.get("sequence")

                    if motif_id and consensus:
                        # Expand IUPAC codes to generate sequences
                        sequences = self._expand_iupac(consensus)
                        for i, seq in enumerate(sequences[:10]):  # Limit expansions
                            if self._is_valid_dna(seq):
                                motif = DNAMotif(
                                    motif_id=f"{motif_id}_{i}",
                                    sequence=seq,
                                    tf_name=entry.get("name"),
                                    tf_class=entry.get("class"),
                                    source="JASPAR",
                                )
                                self.dna_motifs[f"{motif_id}_{i}"] = motif
                                count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} motifs from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_pfm_motifs(self, filepath: Path) -> int:
        """Load motifs from position frequency matrix file"""
        count = 0

        try:
            with open(filepath) as f:
                data = json.load(f)

            for motif_id, pfm_data in data.items():
                # Generate consensus sequence from PFM
                if isinstance(pfm_data, dict) and "A" in pfm_data:
                    consensus = self._pfm_to_consensus(pfm_data)
                    if consensus and self._is_valid_dna(consensus):
                        motif = DNAMotif(
                            motif_id=motif_id,
                            sequence=consensus,
                            source="PFM",
                        )
                        self.dna_motifs[motif_id] = motif
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} motifs from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_binding_sites(self, filepath: Path) -> int:
        """Load binding sites from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline()

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        site_id = parts[0]
                        sequence = parts[1] if len(parts) > 1 else None

                        if sequence and self._is_valid_dna(sequence):
                            motif = DNAMotif(
                                motif_id=site_id,
                                sequence=sequence.upper(),
                                source="binding_sites",
                            )
                            self.dna_motifs[site_id] = motif
                            count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} binding sites from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _expand_iupac(self, sequence: str) -> List[str]:
        """Expand IUPAC ambiguity codes to concrete sequences"""
        if not sequence:
            return []

        sequences = [""]
        for char in sequence.upper():
            if char in IUPAC_DNA:
                new_sequences = []
                for seq in sequences:
                    for nucleotide in IUPAC_DNA[char]:
                        new_sequences.append(seq + nucleotide)
                sequences = new_sequences
            else:
                sequences = [seq + char for seq in sequences]

            # Limit expansion to prevent combinatorial explosion
            if len(sequences) > 100:
                sequences = sequences[:100]
                break

        return sequences

    def _pfm_to_consensus(self, pfm: Dict[str, List[float]]) -> str:
        """Convert position frequency matrix to consensus sequence"""
        consensus = []

        # Get matrix length
        length = len(pfm.get("A", []))

        for i in range(length):
            max_freq = 0
            max_base = "N"
            for base in "ACGT":
                if base in pfm and i < len(pfm[base]):
                    freq = pfm[base][i]
                    if freq > max_freq:
                        max_freq = freq
                        max_base = base
            consensus.append(max_base)

        return "".join(consensus)

    def _is_valid_dna(self, sequence: str) -> bool:
        """Check if DNA sequence is valid"""
        if not sequence:
            return False

        if len(sequence) < self.min_dna_length:
            return False

        if len(sequence) > self.max_dna_length:
            return False

        sequence_upper = sequence.upper()
        invalid_chars = set(sequence_upper) - DNA_NUCLEOTIDES
        if invalid_chars:
            return False

        return True

    def load_interactions(self) -> int:
        """Load protein-DNA interactions from various sources"""
        logger.info("Loading protein-DNA interactions...")

        total_loaded = 0

        # Load from interaction TSV files
        for interaction_file in self.data_dir.glob("**/*interactions*.tsv"):
            count = self._load_interaction_tsv(interaction_file)
            total_loaded += count

        # Load from JASPAR transcription factor files
        for tf_file in self.data_dir.glob("**/*transcription_factors*.tsv"):
            count = self._load_tf_interactions(tf_file)
            total_loaded += count

        logger.info(f"  ✓ Loaded {len(self.interactions):,} positive interactions")
        return len(self.interactions)

    def _load_interaction_tsv(self, filepath: Path) -> int:
        """Load interactions from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header_line = f.readline().strip().lower()
                header = header_line.split("\t")

                # Check if this file might contain DNA interactions
                if "dna" not in filepath.name.lower() and "dna" not in header_line:
                    return 0

                # Find column indices from header
                protein_id_idx = None
                protein_seq_idx = None
                dna_seq_idx = None
                label_idx = None
                source_idx = None

                for i, col in enumerate(header):
                    if col == "protein_id" or col == "uniprot_id":
                        protein_id_idx = i
                    elif col == "protein_seq" or col == "sequence":
                        protein_seq_idx = i
                    elif col == "dna_seq" or col == "dna_sequence":
                        dna_seq_idx = i
                    elif col == "label":
                        label_idx = i
                    elif col == "source":
                        source_idx = i

                # Check if this is a DNA interaction file (must have dna_seq)
                if dna_seq_idx is None:
                    return 0

                # Fall back to positional if header not found
                if protein_id_idx is None:
                    protein_id_idx = 0
                if protein_seq_idx is None:
                    protein_seq_idx = 1
                if label_idx is None:
                    label_idx = 3

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) <= max(protein_id_idx, protein_seq_idx, dna_seq_idx):
                        continue

                    protein_id = parts[protein_id_idx]
                    protein_seq = parts[protein_seq_idx]
                    dna_seq = parts[dna_seq_idx]

                    # Get label
                    try:
                        label = (
                            int(parts[label_idx])
                            if label_idx is not None and len(parts) > label_idx
                            else 1
                        )
                    except ValueError:
                        label = 1  # Default to positive if can't parse

                    # Get source if available
                    source = "interaction_file"
                    if source_idx is not None and len(parts) > source_idx:
                        source = parts[source_idx]

                    if label == 1:
                        # Validate sequences
                        if not self._is_valid_protein(protein_seq):
                            continue
                        if not self._is_valid_dna(dna_seq):
                            continue

                        # Add protein if not exists
                        if protein_id not in self.proteins:
                            self.proteins[protein_id] = ProteinInfo(
                                uniprot_id=protein_id,
                                sequence=protein_seq,
                            )

                        # Create interaction
                        pair_key = (protein_seq, dna_seq.upper())
                        if pair_key not in self.protein_dna_pairs:
                            interaction = InteractionPair(
                                protein_id=protein_id,
                                protein_seq=protein_seq,
                                dna_seq=dna_seq.upper(),
                                label=1,
                                source=source,
                            )
                            self.interactions.append(interaction)
                            self.protein_dna_pairs.add(pair_key)
                            count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} interactions from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _load_tf_interactions(self, filepath: Path) -> int:
        """Load transcription factor interactions from TSV file"""
        count = 0

        try:
            with open(filepath) as f:
                header = f.readline()

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        tf_id = parts[0]
                        uniprot_id = parts[1] if len(parts) > 1 else None
                        motif_id = parts[2] if len(parts) > 2 else None

                        # Get protein sequence
                        protein_seq = None
                        if uniprot_id and uniprot_id in self.proteins:
                            protein_seq = self.proteins[uniprot_id].sequence
                        elif tf_id in self.proteins:
                            protein_seq = self.proteins[tf_id].sequence
                            uniprot_id = tf_id

                        # Get DNA sequence
                        dna_seq = None
                        if motif_id and motif_id in self.dna_motifs:
                            dna_seq = self.dna_motifs[motif_id].sequence

                        if protein_seq and dna_seq:
                            pair_key = (protein_seq, dna_seq)
                            if pair_key not in self.protein_dna_pairs:
                                interaction = InteractionPair(
                                    protein_id=uniprot_id or tf_id,
                                    protein_seq=protein_seq,
                                    dna_seq=dna_seq,
                                    label=1,
                                    source="TF_database",
                                )
                                self.interactions.append(interaction)
                                self.protein_dna_pairs.add(pair_key)
                                count += 1

            if count > 0:
                logger.info(
                    f"    Loaded {count:,} TF interactions from {filepath.name}"
                )

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def generate_negative_samples(self) -> int:
        """Generate negative interaction samples"""
        logger.info("Generating negative samples...")

        # Collect unique proteins and DNA sequences from positive interactions
        unique_proteins = {}
        unique_dna = set()

        for interaction in self.interactions:
            unique_proteins[interaction.protein_seq] = interaction.protein_id
            unique_dna.add(interaction.dna_seq)

        protein_list = list(unique_proteins.keys())
        dna_list = list(unique_dna)

        if not protein_list or not dna_list:
            logger.warning(
                "No proteins or DNA sequences available for negative sampling"
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
            dna_seq = random.choice(dna_list)

            pair_key = (protein_seq, dna_seq)

            if pair_key not in self.protein_dna_pairs:
                protein_id = unique_proteins[protein_seq]
                interaction = InteractionPair(
                    protein_id=protein_id,
                    protein_seq=protein_seq,
                    dna_seq=dna_seq,
                    label=0,
                    source="negative_sampling",
                )
                negative_interactions.append(interaction)
                self.protein_dna_pairs.add(pair_key)

            attempts += 1

        self.interactions.extend(negative_interactions)

        logger.info(f"  ✓ Generated {len(negative_interactions):,} negative samples")
        return len(negative_interactions)

    def augment_data(self) -> int:
        """Augment data with sequence variations"""
        logger.info("Augmenting data...")

        augmented_count = 0

        # DNA reverse complement augmentation for positive samples
        positive_interactions = [i for i in self.interactions if i.label == 1]

        for interaction in positive_interactions[:1000]:  # Limit augmentation
            rc_dna = self._reverse_complement(interaction.dna_seq)

            if rc_dna and rc_dna != interaction.dna_seq:
                pair_key = (interaction.protein_seq, rc_dna)
                if pair_key not in self.protein_dna_pairs:
                    aug_interaction = InteractionPair(
                        protein_id=interaction.protein_id,
                        protein_seq=interaction.protein_seq,
                        dna_seq=rc_dna,
                        label=1,
                        source="augmented_rc",
                    )
                    self.interactions.append(aug_interaction)
                    self.protein_dna_pairs.add(pair_key)
                    augmented_count += 1

        logger.info(f"  ✓ Added {augmented_count:,} augmented samples")
        return augmented_count

    def _reverse_complement(self, sequence: str) -> str:
        """Get reverse complement of DNA sequence"""
        complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
        try:
            return "".join(complement[base] for base in reversed(sequence.upper()))
        except KeyError:
            return ""

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

        # DNA length statistics
        dna_lengths = [len(i.dna_seq) for i in self.interactions]
        features["dna_length_stats"] = {
            "min": int(min(dna_lengths)),
            "max": int(max(dna_lengths)),
            "mean": float(np.mean(dna_lengths)),
            "std": float(np.std(dna_lengths)),
            "median": float(np.median(dna_lengths)),
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
        unique_dna = set(i.dna_seq for i in self.interactions)

        features["unique_stats"] = {
            "unique_proteins": len(unique_proteins),
            "unique_dna_sequences": len(unique_dna),
        }

        # DNA nucleotide composition
        all_dna = "".join(i.dna_seq for i in self.interactions)
        dna_counts = {nt: all_dna.count(nt) for nt in "ACGT"}
        total_nt = len(all_dna)

        features["dna_composition"] = {
            nt: count / total_nt for nt, count in dna_counts.items()
        }

        # GC content
        gc_content = (
            (dna_counts["G"] + dna_counts["C"]) / total_nt if total_nt > 0 else 0
        )
        features["gc_content"] = gc_content

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
            filepath = self.output_dir / f"p2d_{name}.tsv"
            self._save_pairs_tsv(filepath, pairs)

        # Save all interactions
        all_filepath = self.output_dir / "p2d_all.tsv"
        self._save_pairs_tsv(all_filepath, self.interactions)

        # Save unique sequences
        unique_proteins = {}
        unique_dna = {}

        for interaction in self.interactions:
            if interaction.protein_id not in unique_proteins:
                unique_proteins[interaction.protein_id] = interaction.protein_seq
            dna_id = f"DNA_{len(unique_dna)}"
            if interaction.dna_seq not in unique_dna.values():
                unique_dna[dna_id] = interaction.dna_seq

        # Save protein sequences
        proteins_json = {
            pid: {"sequence": seq, "length": len(seq)}
            for pid, seq in unique_proteins.items()
        }
        with open(self.output_dir / "p2d_proteins.json", "w") as f:
            json.dump(proteins_json, f, indent=2)

        # Save DNA sequences
        dna_json = {
            did: {"sequence": seq, "length": len(seq)}
            for did, seq in unique_dna.items()
        }
        with open(self.output_dir / "p2d_dna_sequences.json", "w") as f:
            json.dump(dna_json, f, indent=2)

        # Save features and metadata
        metadata = {
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "min_protein_length": self.min_protein_length,
            "max_protein_length": self.max_protein_length,
            "min_dna_length": self.min_dna_length,
            "max_dna_length": self.max_dna_length,
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

        metadata_filepath = self.output_dir / "p2d_metadata.json"
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save as pickle for faster loading
        pickle_data = {
            "train": [
                (p.protein_id, p.protein_seq, p.dna_seq, p.label, p.score)
                for p in split.train_pairs
            ],
            "val": [
                (p.protein_id, p.protein_seq, p.dna_seq, p.label, p.score)
                for p in split.val_pairs
            ],
            "test": [
                (p.protein_id, p.protein_seq, p.dna_seq, p.label, p.score)
                for p in split.test_pairs
            ],
            "proteins": proteins_json,
            "dna_sequences": dna_json,
            "metadata": metadata,
        }

        pickle_filepath = self.output_dir / "p2d_prepared.pkl"
        with open(pickle_filepath, "wb") as f:
            pickle.dump(pickle_data, f)

        logger.info(f"  ✓ Saved TSV files: p2d_{{train,val,test,all}}.tsv")
        logger.info(f"  ✓ Saved sequences: p2d_proteins.json, p2d_dna_sequences.json")
        logger.info(f"  ✓ Saved metadata: p2d_metadata.json")
        logger.info(f"  ✓ Saved pickle: p2d_prepared.pkl")

    def _save_pairs_tsv(self, filepath: Path, pairs: List[InteractionPair]) -> None:
        """Save interaction pairs to TSV file"""
        with open(filepath, "w") as f:
            f.write("protein_id\tprotein_seq\tdna_seq\tlabel\tscore\tsource\n")

            for pair in pairs:
                f.write(
                    f"{pair.protein_id}\t{pair.protein_seq}\t{pair.dna_seq}\t"
                    f"{pair.label}\t{pair.score:.4f}\t{pair.source}\n"
                )

    def prepare(self) -> Dict[str, Any]:
        """Run full data preparation pipeline"""
        logger.info("=" * 60)
        logger.info("Preparing P2D (Protein-DNA Interaction) Data")
        logger.info("=" * 60)

        # Load data
        self.load_proteins()
        self.load_dna_motifs()
        self.load_interactions()

        # Check if we have enough data
        if len(self.interactions) == 0:
            raise ValueError(
                "No interactions found! Please download real data first using:\n"
                "  python -m download.download_dna --source encode\n"
                "  python -m download.download_dna --source jaspar\n"
                "Synthetic data generation has been disabled."
            )

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
            "num_dna_motifs": len(self.dna_motifs),
            "num_interactions": len(self.interactions),
            "train_size": len(split.train_pairs),
            "val_size": len(split.val_pairs),
            "test_size": len(split.test_pairs),
            "output_dir": str(self.output_dir),
        }

    def _print_summary(self, features: Dict[str, Any]) -> None:
        """Print summary of prepared data"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Data Preparation Summary")
        logger.info("=" * 60)

        stats = features.get("interaction_stats", {})
        protein_stats = features.get("protein_length_stats", {})
        dna_stats = features.get("dna_length_stats", {})
        unique_stats = features.get("unique_stats", {})

        logger.info(f"Total interactions: {stats.get('total_interactions', 0):,}")
        logger.info(f"  - Positive: {stats.get('positive_interactions', 0):,}")
        logger.info(f"  - Negative: {stats.get('negative_interactions', 0):,}")
        logger.info(f"Unique proteins: {unique_stats.get('unique_proteins', 0):,}")
        logger.info(
            f"Unique DNA sequences: {unique_stats.get('unique_dna_sequences', 0):,}"
        )
        logger.info(
            f"Protein length range: {protein_stats.get('min', 0)}-{protein_stats.get('max', 0)}"
        )
        logger.info(
            f"DNA length range: {dna_stats.get('min', 0)}-{dna_stats.get('max', 0)}"
        )
        logger.info(f"GC content: {features.get('gc_content', 0):.2%}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Protein-DNA Interaction data for training",
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
        "--min-dna-length",
        type=int,
        default=6,
        help="Minimum DNA sequence length",
    )

    parser.add_argument(
        "--max-dna-length",
        type=int,
        default=500,
        help="Maximum DNA sequence length",
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
    preparer = PDNADataPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_protein_length=args.min_protein_length,
        max_protein_length=args.max_protein_length,
        min_dna_length=args.min_dna_length,
        max_dna_length=args.max_dna_length,
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
