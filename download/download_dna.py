#!/usr/bin/env python3
"""
Download DNA Data
=================

Unified script for downloading DNA data from multiple sources:
1. JASPAR - Transcription factor binding motifs
2. ENCODE - ChIP-seq binding sites
3. Generated sequences - Synthetic DNA sequences with motifs

This script provides DNA data for protein-DNA interaction models.

Usage:
    python download_dna.py --source jaspar
    python download_dna.py --source jaspar --generate-sequences
    python download_dna.py --generate-random --num-sequences 10000
    python download_dna.py --fasta-file dna_sequences.fasta

Data Sources:
    - JASPAR: Transcription factor binding profiles
    - ENCODE: ChIP-seq binding sites
    - HOCOMOCO: Human and mouse TF binding motifs
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

# JASPAR
JASPAR_API_URL = "https://jaspar.elixir.no/api/v1"

# ENCODE
ENCODE_API_URL = "https://www.encodeproject.org"

# =============================================================================
# DNA VOCABULARY AND UTILITIES
# =============================================================================

DNA_NUCLEOTIDES = ["A", "C", "G", "T"]

# IUPAC ambiguity codes for DNA
IUPAC_CODES = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "U": ["T"],  # Convert U to T
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

# Common TF binding motifs from literature
COMMON_TF_MOTIFS = {
    "TP53": ["RRRCWWGYYY", "RRRCATGYYY", "GGGCATGTCC"],
    "SP1": ["GGGCGG", "GGGGCGGGG", "CCGCCC"],
    "MYC": ["CACGTG", "CACATG", "CATGTG"],
    "CREB1": ["TGACGTCA", "TGACGTMA"],
    "JUN": ["TGACTCA", "TGAGTCA"],
    "FOS": ["TGACTCA", "TGAGTCA"],
    "NFKB1": ["GGGACTTTCC", "GGGRNNYYCC"],
    "RELA": ["GGGACTTTCC", "GGGRNNYYCC"],
    "STAT1": ["TTCNNNGAA", "TTCCNGGAA"],
    "STAT3": ["TTCNNNGAA", "TTCCNGGAA"],
    "E2F1": ["TTTCGCGC", "TTTGGCGC"],
    "YY1": ["CCCATNTT", "GCCATNTT"],
    "ETS1": ["GGAA", "GGAT", "CGGAAGT"],
    "GATA1": ["WGATAR", "AGATAG"],
    "GATA2": ["WGATAR", "AGATAG"],
    "GATA3": ["WGATAR", "AGATAG"],
    "PAX6": ["TTCACGC", "NTTCACGCNSA"],
    "SOX2": ["CATTGTT", "CTTTGTT"],
    "OCT4": ["ATGCAAAT", "ATTTGCAT"],
    "NANOG": ["CATTANN", "AGCCATCA"],
    "HNF4A": ["CAAAGTCCA", "TGGACTTTG"],
    "FOXA1": ["TGTTTAC", "TGTTTGC"],
    "FOXA2": ["TGTTTAC", "TGTTTGC"],
    "CTCF": ["CCGCGNGGNGGCAG", "CCACCAGGGGGCGC"],
    "REST": ["TTCAGCACCATGGACAGCGCC", "NNCAGCACCNNGGACAGNNCC"],
    "RUNX1": ["TGTGGT", "ACCACA"],
    "ERG": ["GGAA", "ACGGAAGT"],
    "SRF": ["CCATATTAGG", "CCWWWWWWGG"],
    "MEF2A": ["CTAWWWWTAG", "YTAWWWWTAR"],
    "CEBPA": ["TTGCGCAA", "ATTGCGCAAT"],
    "CEBPB": ["TTGCGCAA", "ATTGCGCAAT"],
}


def expand_iupac_motif(motif: str) -> str:
    """
    Expand IUPAC ambiguity codes to specific nucleotides

    Args:
        motif: Motif string with IUPAC codes

    Returns:
        Expanded motif with random nucleotide choices
    """
    expanded = ""
    for char in motif.upper():
        if char in IUPAC_CODES:
            expanded += random.choice(IUPAC_CODES[char])
        else:
            expanded += random.choice(DNA_NUCLEOTIDES)
    return expanded


def generate_random_dna(length: int) -> str:
    """
    Generate random DNA sequence

    Args:
        length: Sequence length

    Returns:
        Random DNA sequence
    """
    return "".join(random.choices(DNA_NUCLEOTIDES, k=length))


def generate_dna_with_motif(
    motif: str, min_flank: int = 20, max_flank: int = 50
) -> str:
    """
    Generate DNA sequence containing a specific motif

    Args:
        motif: Binding motif (may contain IUPAC codes)
        min_flank: Minimum flanking sequence length
        max_flank: Maximum flanking sequence length

    Returns:
        DNA sequence containing the motif
    """
    # Expand IUPAC codes
    expanded_motif = expand_iupac_motif(motif)

    # Generate flanking regions
    left_len = random.randint(min_flank, max_flank)
    right_len = random.randint(min_flank, max_flank)

    left_flank = generate_random_dna(left_len)
    right_flank = generate_random_dna(right_len)

    return left_flank + expanded_motif + right_flank


def reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of DNA sequence

    Args:
        sequence: DNA sequence

    Returns:
        Reverse complement
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(sequence.upper()))


def is_valid_dna(sequence: str) -> bool:
    """
    Check if sequence is valid DNA

    Args:
        sequence: DNA sequence

    Returns:
        True if valid DNA sequence
    """
    if not sequence or len(sequence) < 4:
        return False

    valid_chars = set("ACGTN")
    return all(c.upper() in valid_chars for c in sequence)


# =============================================================================
# JASPAR DATABASE
# =============================================================================


def download_jaspar_motifs(
    output_dir: Path,
    collection: str = "CORE",
    tax_group: str = "vertebrates",
    tf_class: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Download transcription factor binding motifs from JASPAR

    Args:
        output_dir: Output directory
        collection: JASPAR collection (CORE, CNE, etc.)
        tax_group: Taxonomic group (vertebrates, plants, etc.)
        tf_class: Optional TF class filter

    Returns:
        Dictionary mapping matrix IDs to motif data
    """
    print("=" * 70)
    print("DOWNLOADING JASPAR MOTIFS")
    print("=" * 70)
    print()

    jaspar_dir = output_dir / "jaspar"
    jaspar_dir.mkdir(parents=True, exist_ok=True)

    # Fetch matrix list
    print(f"Fetching {collection} collection for {tax_group}...")

    params = {
        "collection": collection,
        "tax_group": tax_group,
        "page_size": 1000,
    }

    if tf_class:
        params["tf_class"] = tf_class

    url = f"{JASPAR_API_URL}/matrix/"

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"❌ Error fetching JASPAR data: {e}")
        return {}

    matrices = data.get("results", [])
    print(f"Found {len(matrices)} matrices")

    # Fetch detailed info for each matrix
    motifs: Dict[str, Dict] = {}

    for matrix in tqdm(matrices, desc="Fetching motif details"):
        matrix_id = matrix.get("matrix_id")
        if not matrix_id:
            continue

        try:
            detail_url = f"{JASPAR_API_URL}/matrix/{matrix_id}/"
            response = requests.get(detail_url, timeout=30)
            response.raise_for_status()
            detail = response.json()

            # Extract position frequency matrix
            pfm = detail.get("pfm", {})
            if not pfm:
                continue

            # Convert PFM to consensus sequence
            consensus = ""
            positions = len(pfm.get("A", []))

            for i in range(positions):
                max_freq = 0
                max_base = "N"
                for base in ["A", "C", "G", "T"]:
                    freq = pfm.get(base, [])[i] if i < len(pfm.get(base, [])) else 0
                    if freq > max_freq:
                        max_freq = freq
                        max_base = base
                consensus += max_base

            motifs[matrix_id] = {
                "matrix_id": matrix_id,
                "name": detail.get("name", ""),
                "tf_class": detail.get("class", []),
                "family": detail.get("family", []),
                "species": [s.get("name", "") for s in detail.get("species", [])],
                "uniprot_ids": detail.get("uniprot_ids", []),
                "consensus": consensus,
                "pfm": pfm,
                "source": "JASPAR",
            }

            time.sleep(0.1)  # Rate limiting

        except requests.RequestException:
            continue

    print(f"\n✓ Downloaded {len(motifs):,} motifs")

    # Save motifs
    json_file = jaspar_dir / "jaspar_motifs.json"
    with open(json_file, "w") as f:
        json.dump(motifs, f, indent=2)
    print(f"✓ Saved: {json_file}")

    # Save summary TSV
    tsv_file = jaspar_dir / "jaspar_motifs.tsv"
    with open(tsv_file, "w") as f:
        f.write("matrix_id\tname\tconsensus\tuniprot_ids\ttf_class\tspecies\n")
        for matrix_id, data in motifs.items():
            uniprot_str = ",".join(data.get("uniprot_ids", []))
            class_str = ",".join(data.get("tf_class", []))
            species_str = ",".join(data.get("species", [])[:2])
            f.write(
                f"{matrix_id}\t{data['name']}\t{data['consensus']}\t"
                f"{uniprot_str}\t{class_str}\t{species_str}\n"
            )
    print(f"✓ Saved: {tsv_file}")

    # Save consensus sequences as FASTA
    fasta_file = jaspar_dir / "jaspar_consensus.fasta"
    with open(fasta_file, "w") as f:
        for matrix_id, data in motifs.items():
            f.write(f">{matrix_id}|{data['name']}\n{data['consensus']}\n")
    print(f"✓ Saved: {fasta_file}")

    return motifs


# =============================================================================
# SEQUENCE GENERATION
# =============================================================================


def generate_jaspar_sequences(
    motifs: Dict[str, Dict],
    sequences_per_motif: int = 50,
    include_reverse_complement: bool = True,
) -> List[Dict]:
    """
    Generate DNA sequences containing JASPAR motifs

    Args:
        motifs: Dictionary of JASPAR motifs
        sequences_per_motif: Number of sequences per motif
        include_reverse_complement: Also generate reverse complement sequences

    Returns:
        List of sequence dictionaries
    """
    print("\nGenerating DNA sequences from JASPAR motifs...")

    sequences = []

    for matrix_id, motif_data in tqdm(motifs.items(), desc="Generating sequences"):
        consensus = motif_data.get("consensus", "")
        if not consensus or len(consensus) < 4:
            continue

        tf_name = motif_data.get("name", matrix_id)
        uniprot_ids = motif_data.get("uniprot_ids", [])

        for _ in range(sequences_per_motif):
            # Generate sequence with motif
            dna_seq = generate_dna_with_motif(consensus)
            sequences.append(
                {
                    "sequence": dna_seq,
                    "length": len(dna_seq),
                    "tf_name": tf_name,
                    "matrix_id": matrix_id,
                    "motif": consensus,
                    "uniprot_ids": uniprot_ids,
                    "strand": "+",
                    "source": "jaspar_generated",
                }
            )

            # Optionally add reverse complement
            if include_reverse_complement:
                rc_seq = generate_dna_with_motif(reverse_complement(consensus))
                sequences.append(
                    {
                        "sequence": rc_seq,
                        "length": len(rc_seq),
                        "tf_name": tf_name,
                        "matrix_id": matrix_id,
                        "motif": reverse_complement(consensus),
                        "uniprot_ids": uniprot_ids,
                        "strand": "-",
                        "source": "jaspar_generated",
                    }
                )

    print(f"✓ Generated {len(sequences):,} sequences")
    return sequences


def generate_common_tf_sequences(
    sequences_per_tf: int = 100,
    include_reverse_complement: bool = True,
) -> List[Dict]:
    """
    Generate DNA sequences using common literature-curated TF motifs

    Args:
        sequences_per_tf: Number of sequences per TF
        include_reverse_complement: Include reverse complement

    Returns:
        List of sequence dictionaries
    """
    print("\nGenerating sequences from common TF motifs...")

    sequences = []

    for tf_name, motifs in tqdm(COMMON_TF_MOTIFS.items(), desc="Generating"):
        seqs_per_motif = sequences_per_tf // len(motifs)

        for motif in motifs:
            for _ in range(seqs_per_motif):
                dna_seq = generate_dna_with_motif(motif)
                sequences.append(
                    {
                        "sequence": dna_seq,
                        "length": len(dna_seq),
                        "tf_name": tf_name,
                        "motif": motif,
                        "strand": "+",
                        "source": "common_motifs",
                    }
                )

                if include_reverse_complement:
                    rc_seq = generate_dna_with_motif(reverse_complement(motif))
                    sequences.append(
                        {
                            "sequence": rc_seq,
                            "length": len(rc_seq),
                            "tf_name": tf_name,
                            "motif": reverse_complement(motif),
                            "strand": "-",
                            "source": "common_motifs",
                        }
                    )

    print(f"✓ Generated {len(sequences):,} sequences")
    return sequences


def generate_random_sequences(
    num_sequences: int = 10000,
    min_length: int = 50,
    max_length: int = 150,
) -> List[Dict]:
    """
    Generate random DNA sequences (for negative sampling)

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of sequence dictionaries
    """
    print(f"\nGenerating {num_sequences:,} random DNA sequences...")

    sequences = []

    for i in tqdm(range(num_sequences), desc="Generating random sequences"):
        length = random.randint(min_length, max_length)
        dna_seq = generate_random_dna(length)
        sequences.append(
            {
                "sequence": dna_seq,
                "length": length,
                "tf_name": None,
                "motif": None,
                "source": "random",
            }
        )

    print(f"✓ Generated {len(sequences):,} random sequences")
    return sequences


# =============================================================================
# FILE I/O
# =============================================================================


def load_fasta_file(filepath: Path) -> List[Dict]:
    """
    Load DNA sequences from FASTA file

    Args:
        filepath: Path to FASTA file

    Returns:
        List of sequence dictionaries
    """
    print(f"Loading sequences from {filepath}...")

    sequences = []
    current_id = None
    current_seq: List[str] = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence
                if current_id and current_seq:
                    seq = "".join(current_seq).upper()
                    if is_valid_dna(seq):
                        sequences.append(
                            {
                                "sequence": seq,
                                "length": len(seq),
                                "seq_id": current_id,
                                "source": "fasta",
                            }
                        )

                # Start new sequence
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last sequence
        if current_id and current_seq:
            seq = "".join(current_seq).upper()
            if is_valid_dna(seq):
                sequences.append(
                    {
                        "sequence": seq,
                        "length": len(seq),
                        "seq_id": current_id,
                        "source": "fasta",
                    }
                )

    print(f"✓ Loaded {len(sequences):,} sequences")
    return sequences


def save_sequences(
    sequences: List[Dict],
    output_dir: Path,
    motifs: Optional[Dict] = None,
) -> None:
    """
    Save DNA sequence data

    Args:
        sequences: List of sequence dictionaries
        output_dir: Output directory
        motifs: Optional motif data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as FASTA
    fasta_file = output_dir / "dna_sequences.fasta"
    with open(fasta_file, "w") as f:
        for i, seq_data in enumerate(sequences):
            seq_id = seq_data.get("seq_id", f"dna_{i}")
            tf_name = seq_data.get("tf_name", "")
            motif = seq_data.get("motif", "")

            header = f">{seq_id}"
            if tf_name:
                header += f"|tf={tf_name}"
            if motif:
                header += f"|motif={motif}"

            f.write(f"{header}\n{seq_data['sequence']}\n")

    print(f"✓ Saved FASTA: {fasta_file}")

    # Save as JSON
    json_file = output_dir / "dna_sequences.json"
    with open(json_file, "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"✓ Saved JSON: {json_file}")

    # Save as TSV
    tsv_file = output_dir / "dna_sequences.tsv"
    with open(tsv_file, "w") as f:
        f.write("seq_id\tsequence\tlength\ttf_name\tmotif\tstrand\tsource\n")
        for i, seq_data in enumerate(sequences):
            seq_id = seq_data.get("seq_id", f"dna_{i}")
            f.write(
                f"{seq_id}\t{seq_data['sequence']}\t{seq_data['length']}\t"
                f"{seq_data.get('tf_name', '')}\t{seq_data.get('motif', '')}\t"
                f"{seq_data.get('strand', '')}\t{seq_data.get('source', '')}\n"
            )
    print(f"✓ Saved TSV: {tsv_file}")

    # Save TF-specific files
    if any(seq.get("tf_name") for seq in sequences):
        tf_dir = output_dir / "by_tf"
        tf_dir.mkdir(exist_ok=True)

        # Group sequences by TF
        tf_sequences: Dict[str, List[Dict]] = {}
        for seq_data in sequences:
            tf = seq_data.get("tf_name")
            if tf:
                if tf not in tf_sequences:
                    tf_sequences[tf] = []
                tf_sequences[tf].append(seq_data)

        # Save per-TF files
        for tf, seqs in tf_sequences.items():
            tf_file = tf_dir / f"{tf}_sequences.fasta"
            with open(tf_file, "w") as f:
                for i, seq_data in enumerate(seqs):
                    f.write(f">{tf}_{i}|motif={seq_data.get('motif', '')}\n")
                    f.write(f"{seq_data['sequence']}\n")

        print(f"✓ Saved {len(tf_sequences)} TF-specific files to {tf_dir}")


# =============================================================================
# METADATA AND SUMMARY
# =============================================================================


def save_metadata(
    output_dir: Path,
    source: str,
    sequence_count: int,
    tf_count: int = 0,
    motif_count: int = 0,
) -> None:
    """Save download metadata"""
    output_file = output_dir / "download_info.json"

    info = {
        "source": source,
        "sequence_count": sequence_count,
        "tf_count": tf_count,
        "motif_count": motif_count,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved metadata: {output_file}")


def print_summary(
    output_dir: Path,
    sequences: List[Dict],
    motifs: Optional[Dict] = None,
) -> None:
    """Print download summary"""
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Sequences generated: {len(sequences):,}")

    if motifs:
        print(f"TFs with motifs: {len(motifs):,}")

    # Count by source
    sources = {}
    for seq in sequences:
        src = seq.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\nSequences by source:")
    for src, count in sorted(sources.items()):
        print(f"  • {src}: {count:,}")

    # Count unique TFs
    tfs = set(seq.get("tf_name") for seq in sequences if seq.get("tf_name"))
    if tfs:
        print(f"\nUnique TFs: {len(tfs)}")

    print()
    print("Files created:")
    print("  • dna_sequences.fasta - DNA sequences (FASTA)")
    print("  • dna_sequences.json - Full data (JSON)")
    print("  • dna_sequences.tsv - Summary (TSV)")
    if motifs:
        print("  • jaspar/ - JASPAR database files")
    if tfs:
        print("  • by_tf/ - TF-specific sequence files")
    print("  • download_info.json - Metadata")
    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download DNA Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_dna.py --source jaspar
    python download_dna.py --source jaspar --generate-sequences
    python download_dna.py --source common --sequences-per-tf 100
    python download_dna.py --generate-random --num-sequences 10000
    python download_dna.py --fasta-file dna_sequences.fasta

Sources:
    jaspar - JASPAR database (TF binding motifs)
    common - Common literature-curated TF motifs
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="jaspar",
        choices=["jaspar", "common"],
        help="Data source for TF motifs (default: jaspar)",
    )

    parser.add_argument(
        "--generate-sequences",
        action="store_true",
        help="Generate DNA sequences containing binding motifs",
    )

    parser.add_argument(
        "--sequences-per-motif",
        type=int,
        default=50,
        help="Number of sequences to generate per motif (default: 50)",
    )

    parser.add_argument(
        "--sequences-per-tf",
        type=int,
        default=100,
        help="Number of sequences to generate per TF for common motifs (default: 100)",
    )

    parser.add_argument(
        "--generate-random",
        action="store_true",
        help="Generate random DNA sequences (for negative sampling)",
    )

    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10000,
        help="Number of random sequences to generate (default: 10000)",
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum sequence length (default: 50)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Maximum sequence length (default: 150)",
    )

    parser.add_argument(
        "--fasta-file",
        type=str,
        help="Load DNA sequences from FASTA file",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="CORE",
        help="JASPAR collection (default: CORE)",
    )

    parser.add_argument(
        "--tax-group",
        type=str,
        default="vertebrates",
        help="JASPAR taxonomic group (default: vertebrates)",
    )

    parser.add_argument(
        "--include-reverse-complement",
        action="store_true",
        default=True,
        help="Include reverse complement sequences (default: True)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dna",
        help="Output directory (default: data/dna)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 70)
    print("DNA DATA DOWNLOAD")
    print("=" * 70)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    sequences: List[Dict] = []
    motifs: Optional[Dict] = None

    # Load from FASTA file if specified
    if args.fasta_file:
        sequences = load_fasta_file(Path(args.fasta_file))

    # Generate random sequences
    elif args.generate_random:
        sequences = generate_random_sequences(
            num_sequences=args.num_sequences,
            min_length=args.min_length,
            max_length=args.max_length,
        )

    # Download from JASPAR
    elif args.source == "jaspar":
        motifs = download_jaspar_motifs(
            output_dir,
            collection=args.collection,
            tax_group=args.tax_group,
        )

        if args.generate_sequences and motifs:
            sequences = generate_jaspar_sequences(
                motifs,
                sequences_per_motif=args.sequences_per_motif,
                include_reverse_complement=args.include_reverse_complement,
            )

    # Use common literature motifs
    elif args.source == "common":
        sequences = generate_common_tf_sequences(
            sequences_per_tf=args.sequences_per_tf,
            include_reverse_complement=args.include_reverse_complement,
        )

    # Save sequences if generated
    if sequences:
        save_sequences(sequences, output_dir, motifs)

        # Save metadata
        save_metadata(
            output_dir=output_dir,
            source=args.source if not args.fasta_file else "fasta",
            sequence_count=len(sequences),
            tf_count=len(motifs) if motifs else len(COMMON_TF_MOTIFS),
            motif_count=len(motifs)
            if motifs
            else sum(len(v) for v in COMMON_TF_MOTIFS.values()),
        )

    elif motifs:
        # Just downloaded motifs, no sequences generated
        save_metadata(
            output_dir=output_dir,
            source=args.source,
            sequence_count=0,
            tf_count=len(motifs),
            motif_count=len(motifs),
        )

    # Print summary
    print_summary(output_dir, sequences, motifs)

    print("Done!")
    print()
    print("This DNA data can be used with:")
    print("  • download_p2d.py - Protein-DNA interactions")


if __name__ == "__main__":
    main()
