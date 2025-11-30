#!/usr/bin/env python3
"""
Prepare Protein-Protein Interaction (P2P) Data for Training

This script processes raw downloaded protein data and prepares it for
training the P2P interaction prediction model.

Steps:
1. Load protein sequences from UniProt/STRING downloads
2. Load interaction data from STRING database
3. Filter and validate sequences
4. Generate negative samples using intelligent sampling strategies
5. Split data into train/val/test sets
6. Compute sequence features and statistics
7. Save prepared data in training-ready format
"""

import argparse
import gzip
import hashlib
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared" / "p2p"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ProteinInfo:
    """Store protein information"""

    uniprot_id: str
    sequence: str
    string_id: Optional[str] = None
    gene_name: Optional[str] = None
    description: Optional[str] = None
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


@dataclass
class InteractionPair:
    """Store interaction pair information"""

    protein1_id: str
    protein2_id: str
    score: float
    label: int  # 1 for positive, 0 for negative
    source: str = "STRING"
    evidence_types: List[str] = field(default_factory=list)


@dataclass
class DataSplit:
    """Store data split information"""

    train_pairs: List[InteractionPair]
    val_pairs: List[InteractionPair]
    test_pairs: List[InteractionPair]

    @property
    def total_pairs(self) -> int:
        return len(self.train_pairs) + len(self.val_pairs) + len(self.test_pairs)


class PPIDataPreparer:
    """Prepare protein-protein interaction data for training"""

    # Standard amino acids
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    # Non-standard but acceptable amino acids
    EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        min_sequence_length: int = 50,
        max_sequence_length: int = 2000,
        min_interaction_score: float = 0.4,
        negative_ratio: float = 1.0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.min_interaction_score = min_interaction_score
        self.negative_ratio = negative_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Data containers
        self.proteins: Dict[str, ProteinInfo] = {}
        self.id_mapping: Dict[str, str] = {}  # STRING -> UniProt
        self.interactions: List[InteractionPair] = []
        self.protein_degrees: Dict[str, int] = defaultdict(int)

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_sequences(self) -> int:
        """Load protein sequences from various sources"""
        logger.info("Loading protein sequences...")

        total_loaded = 0

        # Try loading from UniProt FASTA files
        fasta_patterns = [
            "**/*.fasta",
            "**/*.fasta.gz",
            "**/*sequences*.txt",
            "**/*sequences*.txt.gz",
        ]

        for pattern in fasta_patterns:
            for fasta_file in self.data_dir.glob(pattern):
                count = self._load_fasta(fasta_file)
                total_loaded += count

        # Try loading from JSON files
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
                            if self._is_valid_sequence(seq):
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
                            # UniProt format: >sp|P12345|PROT_HUMAN Description
                            current_id = parts[1]
                            current_desc = parts[2] if len(parts) > 2 else None
                        else:
                            # Simple format: >PROTEIN_ID Description
                            parts = header.split()
                            current_id = parts[0]
                            current_desc = (
                                " ".join(parts[1:]) if len(parts) > 1 else None
                            )

                        current_seq = []
                    else:
                        current_seq.append(line)

                # Don't forget the last protein
                if current_id and current_seq:
                    seq = "".join(current_seq)
                    if self._is_valid_sequence(seq):
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

            # Handle different JSON formats
            proteins_list = data if isinstance(data, list) else data.get("proteins", [])

            for entry in proteins_list:
                if isinstance(entry, dict):
                    protein_id = (
                        entry.get("uniprot_id")
                        or entry.get("id")
                        or entry.get("accession")
                    )
                    sequence = entry.get("sequence")

                    if protein_id and sequence and self._is_valid_sequence(sequence):
                        protein = ProteinInfo(
                            uniprot_id=protein_id,
                            sequence=sequence,
                            gene_name=entry.get("gene_name"),
                            description=entry.get("description"),
                        )
                        self.proteins[protein_id] = protein
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} proteins from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _is_valid_sequence(self, sequence: str) -> bool:
        """Check if sequence is valid for training"""
        if not sequence:
            return False

        if len(sequence) < self.min_sequence_length:
            return False

        if len(sequence) > self.max_sequence_length:
            return False

        # Check for valid amino acids (allow some flexibility)
        sequence_upper = sequence.upper()
        invalid_chars = set(sequence_upper) - self.EXTENDED_AA
        if invalid_chars:
            return False

        return True

    def load_id_mapping(self) -> int:
        """Load STRING to UniProt ID mappings"""
        logger.info("Loading ID mappings...")

        mapping_files = list(self.data_dir.glob("**/*mapping*.tsv")) + list(
            self.data_dir.glob("**/*mapping*.txt")
        )

        for mapping_file in mapping_files:
            try:
                with open(mapping_file) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            string_id = parts[0]
                            uniprot_id = parts[1]
                            if uniprot_id in self.proteins:
                                self.id_mapping[string_id] = uniprot_id
                                self.proteins[uniprot_id].string_id = string_id

                logger.info(f"    Loaded mappings from {mapping_file.name}")

            except Exception as e:
                logger.warning(f"    Failed to load {mapping_file}: {e}")

        # Also create reverse mapping for proteins without STRING IDs
        for uniprot_id in self.proteins:
            if uniprot_id not in self.id_mapping.values():
                self.id_mapping[uniprot_id] = uniprot_id

        logger.info(f"  ✓ Loaded {len(self.id_mapping):,} ID mappings")
        return len(self.id_mapping)

    def load_interactions(self) -> int:
        """Load protein-protein interactions from STRING or other sources"""
        logger.info("Loading interactions...")

        total_interactions = 0

        # Try STRING links files
        links_files = (
            list(self.data_dir.glob("**/*links*.txt"))
            + list(self.data_dir.glob("**/*links*.txt.gz"))
            + list(self.data_dir.glob("**/*interactions*.tsv"))
        )

        for links_file in links_files:
            count = self._load_string_links(links_file)
            total_interactions += count

        # Calculate protein degrees
        for interaction in self.interactions:
            self.protein_degrees[interaction.protein1_id] += 1
            self.protein_degrees[interaction.protein2_id] += 1

        logger.info(f"  ✓ Loaded {len(self.interactions):,} positive interactions")
        return len(self.interactions)

    def _load_string_links(self, filepath: Path) -> int:
        """Load interactions from STRING links file"""
        count = 0
        seen_pairs = set()

        try:
            open_func = gzip.open if filepath.suffix == ".gz" else open
            mode = "rt" if filepath.suffix == ".gz" else "r"

            with open_func(filepath, mode) as f:
                header = f.readline()  # Skip header

                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        protein1 = parts[0]
                        protein2 = parts[1]

                        try:
                            score = float(parts[2])
                            # Normalize score to 0-1 range if needed
                            if score > 1:
                                score = score / 1000.0
                        except ValueError:
                            continue

                        if score < self.min_interaction_score:
                            continue

                        # Map to UniProt IDs
                        uniprot1 = self._get_uniprot_id(protein1)
                        uniprot2 = self._get_uniprot_id(protein2)

                        if not uniprot1 or not uniprot2:
                            continue

                        if (
                            uniprot1 not in self.proteins
                            or uniprot2 not in self.proteins
                        ):
                            continue

                        # Avoid duplicates
                        pair_key = tuple(sorted([uniprot1, uniprot2]))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        interaction = InteractionPair(
                            protein1_id=uniprot1,
                            protein2_id=uniprot2,
                            score=score,
                            label=1,
                            source="STRING",
                        )
                        self.interactions.append(interaction)
                        count += 1

            if count > 0:
                logger.info(f"    Loaded {count:,} interactions from {filepath.name}")

        except Exception as e:
            logger.warning(f"    Failed to load {filepath}: {e}")

        return count

    def _get_uniprot_id(self, protein_id: str) -> Optional[str]:
        """Get UniProt ID from various ID formats"""
        # Direct match
        if protein_id in self.proteins:
            return protein_id

        # Check mapping
        if protein_id in self.id_mapping:
            return self.id_mapping[protein_id]

        # Try extracting from STRING format (e.g., "9606.ENSP00000000001")
        if "." in protein_id:
            parts = protein_id.split(".")
            for part in parts:
                if part in self.id_mapping:
                    return self.id_mapping[part]
                if part in self.proteins:
                    return part

        return None

    def generate_negative_samples(self) -> int:
        """Generate negative interaction samples using intelligent sampling"""
        logger.info("Generating negative samples...")

        # Build positive pair set
        positive_pairs = set()
        for interaction in self.interactions:
            pair = tuple(sorted([interaction.protein1_id, interaction.protein2_id]))
            positive_pairs.add(pair)

        # Get proteins involved in positive interactions
        interacting_proteins = set()
        for interaction in self.interactions:
            interacting_proteins.add(interaction.protein1_id)
            interacting_proteins.add(interaction.protein2_id)

        protein_list = list(interacting_proteins)

        # Calculate number of negatives to generate
        num_negatives = int(len(self.interactions) * self.negative_ratio)

        logger.info(f"  Generating {num_negatives:,} negative samples...")

        negative_interactions = []
        attempts = 0
        max_attempts = num_negatives * 20

        # Use degree-aware negative sampling
        # Proteins with higher degree are more likely to be sampled
        degrees = np.array([self.protein_degrees.get(p, 1) for p in protein_list])
        sampling_weights = degrees / degrees.sum()

        while len(negative_interactions) < num_negatives and attempts < max_attempts:
            # Sample with probability proportional to degree
            idx1, idx2 = np.random.choice(
                len(protein_list), size=2, replace=False, p=sampling_weights
            )
            protein1 = protein_list[idx1]
            protein2 = protein_list[idx2]

            pair_key = tuple(sorted([protein1, protein2]))

            if pair_key not in positive_pairs:
                interaction = InteractionPair(
                    protein1_id=protein1,
                    protein2_id=protein2,
                    score=0.0,
                    label=0,
                    source="negative_sampling",
                )
                negative_interactions.append(interaction)
                positive_pairs.add(pair_key)  # Prevent duplicate negatives

            attempts += 1

        self.interactions.extend(negative_interactions)

        logger.info(f"  ✓ Generated {len(negative_interactions):,} negative samples")
        return len(negative_interactions)

    def compute_sequence_features(self) -> Dict[str, Any]:
        """Compute sequence-level features and statistics"""
        logger.info("Computing sequence features...")

        features = {}

        # Get proteins involved in interactions
        involved_proteins = set()
        for interaction in self.interactions:
            involved_proteins.add(interaction.protein1_id)
            involved_proteins.add(interaction.protein2_id)

        # Compute length statistics
        lengths = [
            self.proteins[p].length for p in involved_proteins if p in self.proteins
        ]

        features["length_stats"] = {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "median": float(np.median(lengths)),
        }

        # Compute amino acid composition
        aa_counts = defaultdict(int)
        total_residues = 0

        for protein_id in involved_proteins:
            if protein_id in self.proteins:
                seq = self.proteins[protein_id].sequence.upper()
                for aa in seq:
                    aa_counts[aa] += 1
                    total_residues += 1

        features["aa_composition"] = {
            aa: count / total_residues for aa, count in aa_counts.items()
        }

        # Compute interaction statistics
        positive_count = sum(1 for i in self.interactions if i.label == 1)
        negative_count = sum(1 for i in self.interactions if i.label == 0)

        features["interaction_stats"] = {
            "total_proteins": len(involved_proteins),
            "total_interactions": len(self.interactions),
            "positive_interactions": positive_count,
            "negative_interactions": negative_count,
            "positive_ratio": positive_count / len(self.interactions)
            if self.interactions
            else 0,
        }

        # Score distribution for positive interactions
        positive_scores = [i.score for i in self.interactions if i.label == 1]
        if positive_scores:
            features["score_stats"] = {
                "min": float(min(positive_scores)),
                "max": float(max(positive_scores)),
                "mean": float(np.mean(positive_scores)),
                "std": float(np.std(positive_scores)),
            }

        logger.info(f"  ✓ Computed features for {len(involved_proteins):,} proteins")
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
            filepath = self.output_dir / f"p2p_{name}.tsv"
            self._save_pairs_tsv(filepath, pairs)

        # Save all interactions as single file
        all_filepath = self.output_dir / "p2p_all.tsv"
        self._save_pairs_tsv(all_filepath, self.interactions)

        # Save sequences as FASTA
        involved_proteins = set()
        for interaction in self.interactions:
            involved_proteins.add(interaction.protein1_id)
            involved_proteins.add(interaction.protein2_id)

        fasta_filepath = self.output_dir / "p2p_sequences.fasta"
        self._save_sequences_fasta(fasta_filepath, involved_proteins)

        # Save sequences as JSON (easier to load)
        sequences_json = {
            protein_id: {
                "sequence": self.proteins[protein_id].sequence,
                "length": self.proteins[protein_id].length,
                "gene_name": self.proteins[protein_id].gene_name,
            }
            for protein_id in involved_proteins
            if protein_id in self.proteins
        }

        json_filepath = self.output_dir / "p2p_sequences.json"
        with open(json_filepath, "w") as f:
            json.dump(sequences_json, f, indent=2)

        # Save features and metadata
        metadata = {
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "min_sequence_length": self.min_sequence_length,
            "max_sequence_length": self.max_sequence_length,
            "min_interaction_score": self.min_interaction_score,
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

        metadata_filepath = self.output_dir / "p2p_metadata.json"
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save as pickle for faster loading
        pickle_data = {
            "train": [
                (p.protein1_id, p.protein2_id, p.label, p.score)
                for p in split.train_pairs
            ],
            "val": [
                (p.protein1_id, p.protein2_id, p.label, p.score)
                for p in split.val_pairs
            ],
            "test": [
                (p.protein1_id, p.protein2_id, p.label, p.score)
                for p in split.test_pairs
            ],
            "sequences": sequences_json,
            "metadata": metadata,
        }

        pickle_filepath = self.output_dir / "p2p_prepared.pkl"
        with open(pickle_filepath, "wb") as f:
            pickle.dump(pickle_data, f)

        logger.info(f"  ✓ Saved TSV files: p2p_{{train,val,test,all}}.tsv")
        logger.info(f"  ✓ Saved sequences: p2p_sequences.{{fasta,json}}")
        logger.info(f"  ✓ Saved metadata: p2p_metadata.json")
        logger.info(f"  ✓ Saved pickle: p2p_prepared.pkl")

    def _save_pairs_tsv(self, filepath: Path, pairs: List[InteractionPair]) -> None:
        """Save interaction pairs to TSV file"""
        with open(filepath, "w") as f:
            f.write(
                "protein1_id\tprotein2_id\tprotein1_seq\tprotein2_seq\tlabel\tscore\tsource\n"
            )

            for pair in pairs:
                seq1 = self.proteins.get(pair.protein1_id, ProteinInfo("", "")).sequence
                seq2 = self.proteins.get(pair.protein2_id, ProteinInfo("", "")).sequence

                f.write(
                    f"{pair.protein1_id}\t{pair.protein2_id}\t{seq1}\t{seq2}\t"
                    f"{pair.label}\t{pair.score:.4f}\t{pair.source}\n"
                )

    def _save_sequences_fasta(self, filepath: Path, protein_ids: Set[str]) -> None:
        """Save sequences to FASTA file"""
        with open(filepath, "w") as f:
            for protein_id in sorted(protein_ids):
                if protein_id in self.proteins:
                    protein = self.proteins[protein_id]
                    f.write(f">{protein_id}")
                    if protein.gene_name:
                        f.write(f" | {protein.gene_name}")
                    if protein.description:
                        f.write(f" | {protein.description[:100]}")
                    f.write("\n")

                    # Write sequence in lines of 80 characters
                    seq = protein.sequence
                    for i in range(0, len(seq), 80):
                        f.write(f"{seq[i : i + 80]}\n")

    def prepare(self) -> Dict[str, Any]:
        """Run full data preparation pipeline"""
        logger.info("=" * 60)
        logger.info("Preparing P2P (Protein-Protein Interaction) Data")
        logger.info("=" * 60)

        # Load data
        self.load_sequences()
        self.load_id_mapping()
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
        features = self.compute_sequence_features()

        # Split data
        split = self.split_data()

        # Save prepared data
        self.save_prepared_data(split, features)

        # Print summary
        self._print_summary(features)

        return {
            "num_proteins": len(self.proteins),
            "num_interactions": len(self.interactions),
            "train_size": len(split.train_pairs),
            "val_size": len(split.val_pairs),
            "test_size": len(split.test_pairs),
            "output_dir": str(self.output_dir),
        }

    def _create_synthetic_data(self) -> None:
        """Create synthetic data for testing when no real data is available"""
        logger.info("Creating synthetic protein data for testing...")

        # Generate synthetic proteins
        for i in range(100):
            protein_id = f"SYNTH_{i:04d}"
            length = random.randint(100, 500)
            sequence = "".join(random.choices(list(self.AMINO_ACIDS), k=length))

            self.proteins[protein_id] = ProteinInfo(
                uniprot_id=protein_id,
                sequence=sequence,
                gene_name=f"GENE_{i}",
            )

        # Generate synthetic interactions
        protein_list = list(self.proteins.keys())
        for _ in range(200):
            p1, p2 = random.sample(protein_list, 2)
            score = random.uniform(0.5, 1.0)

            self.interactions.append(
                InteractionPair(
                    protein1_id=p1,
                    protein2_id=p2,
                    score=score,
                    label=1,
                    source="synthetic",
                )
            )

        logger.info(f"  ✓ Created {len(self.proteins)} synthetic proteins")
        logger.info(f"  ✓ Created {len(self.interactions)} synthetic interactions")

    def _print_summary(self, features: Dict[str, Any]) -> None:
        """Print summary of prepared data"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Data Preparation Summary")
        logger.info("=" * 60)

        stats = features.get("interaction_stats", {})
        length_stats = features.get("length_stats", {})

        logger.info(f"Total proteins: {stats.get('total_proteins', 0):,}")
        logger.info(f"Total interactions: {stats.get('total_interactions', 0):,}")
        logger.info(f"  - Positive: {stats.get('positive_interactions', 0):,}")
        logger.info(f"  - Negative: {stats.get('negative_interactions', 0):,}")
        logger.info(
            f"Sequence length range: {length_stats.get('min', 0)}-{length_stats.get('max', 0)}"
        )
        logger.info(f"Mean sequence length: {length_stats.get('mean', 0):.1f}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Protein-Protein Interaction data for training",
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
        "--min-sequence-length",
        type=int,
        default=50,
        help="Minimum protein sequence length",
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2000,
        help="Maximum protein sequence length",
    )

    parser.add_argument(
        "--min-interaction-score",
        type=float,
        default=0.4,
        help="Minimum interaction score threshold (0-1)",
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
    preparer = PPIDataPreparer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        min_interaction_score=args.min_interaction_score,
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
