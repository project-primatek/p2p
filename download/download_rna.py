#!/usr/bin/env python3
"""
Download RNA Data
=================

Unified script for downloading RNA data from multiple sources:
1. ATtRACT - RNA-binding protein motifs database
2. RNAcentral - Comprehensive RNA sequence database
3. RBPDB - RNA-binding protein database
4. Generated sequences - Synthetic RNA sequences with motifs

This script provides RNA data for protein-RNA interaction models.

Usage:
    python download_rna.py --source attract
    python download_rna.py --source attract --generate-sequences
    python download_rna.py --generate-random --num-sequences 10000
    python download_rna.py --fasta-file rna_sequences.fasta

Data Sources:
    - ATtRACT: RNA-binding protein motifs and binding preferences
    - RNAcentral: Non-coding RNA sequences
    - RBPDB: Experimentally determined RNA-binding specificities
"""

import argparse
import zipfile
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

# ATtRACT database
ATTRACT_URL = "https://attract.cnic.es/attract/static/ATtRACT.zip"

# RNAcentral
RNACENTRAL_API_URL = "https://rnacentral.org/api/v1"

# =============================================================================
# RNA VOCABULARY AND UTILITIES
# =============================================================================

RNA_NUCLEOTIDES = ["A", "C", "G", "U"]

# IUPAC ambiguity codes for RNA
IUPAC_CODES = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "U": ["U"],
    "T": ["U"],  # Convert T to U
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

# Common RBP binding motifs from literature
COMMON_RBP_MOTIFS = {
    "HNRNPA1": ["UAGGGA", "UAGGGU", "UAGGG"],
    "HNRNPC": ["UUUUU", "UUUUUU", "UUUUUUU"],
    "SRSF1": ["GGAGGAA", "GAAGAA", "RGAAGAAC"],
    "SRSF2": ["GGNG", "CCNG", "SSNG"],
    "TIA1": ["UUUUU", "UUUUUU", "WWUUUW"],
    "TIAL1": ["UUUUU", "UUUUUU"],
    "ELAVL1": ["AUUUA", "UAUUUAU", "NNUUNNUUU"],
    "RBFOX2": ["UGCAUG", "GCAUG", "WGCAUGM"],
    "PTBP1": ["UCUU", "CUCUCU", "YYUCUUY"],
    "QKI": ["ACUAA", "CUAAC", "NACUAAY"],
    "FUS": ["GGUG", "GUGGU", "GGUGU"],
    "TARDBP": ["UGUGU", "GUGU", "UGUGUGU"],
    "IGF2BP1": ["CAUH", "ACACCC", "CAUCAU"],
    "IGF2BP2": ["CAUH", "ACACCC"],
    "IGF2BP3": ["CAUH", "ACACCC"],
    "KHDRBS1": ["UAAA", "UUAA", "UWAA"],
    "RBM22": ["GGUG", "GUGG"],
    "U2AF2": ["UUUUU", "UUUUUU", "UUUUUUUU"],
    "SF3B4": ["UGUGUG", "UGUG"],
    "PRPF8": ["GU", "AG"],
    "YTHDC1": ["GGACU", "GGACA", "DRACH"],
    "YTHDF1": ["GGACU", "DRACH"],
    "YTHDF2": ["GGACU", "DRACH"],
    "METTL3": ["DRACH", "RRACH"],
    "FMR1": ["WGGA", "GACG", "DWGGADWGGA"],
    "CELF1": ["UGUU", "UGUGU"],
    "NOVA1": ["YCAY", "UCAU"],
    "NOVA2": ["YCAY", "UCAU"],
    "MBNL1": ["YGCY", "UGCU"],
    "PCBP1": ["CCCC", "CCCCC"],
    "PCBP2": ["CCCC", "CCCCC"],
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
            expanded += random.choice(RNA_NUCLEOTIDES)
    return expanded


def generate_random_rna(length: int) -> str:
    """
    Generate random RNA sequence

    Args:
        length: Sequence length

    Returns:
        Random RNA sequence
    """
    return "".join(random.choices(RNA_NUCLEOTIDES, k=length))


def generate_rna_with_motif(
    motif: str, min_flank: int = 15, max_flank: int = 40
) -> str:
    """
    Generate RNA sequence containing a specific motif

    Args:
        motif: Binding motif (may contain IUPAC codes)
        min_flank: Minimum flanking sequence length
        max_flank: Maximum flanking sequence length

    Returns:
        RNA sequence containing the motif
    """
    # Expand IUPAC codes
    expanded_motif = expand_iupac_motif(motif)

    # Generate flanking regions
    left_len = random.randint(min_flank, max_flank)
    right_len = random.randint(min_flank, max_flank)

    left_flank = generate_random_rna(left_len)
    right_flank = generate_random_rna(right_len)

    return left_flank + expanded_motif + right_flank


def is_valid_rna(sequence: str) -> bool:
    """
    Check if sequence is valid RNA

    Args:
        sequence: RNA sequence

    Returns:
        True if valid RNA sequence
    """
    if not sequence or len(sequence) < 4:
        return False

    valid_chars = set("ACGUT")
    return all(c.upper() in valid_chars for c in sequence)


# =============================================================================
# ATTRACT DATABASE
# =============================================================================


def download_attract_database(output_dir: Path) -> Dict[str, List[Dict]]:
    """
    Download and parse the ATtRACT database

    Args:
        output_dir: Output directory

    Returns:
        Dictionary mapping RBP names to binding motifs
    """
    print("=" * 70)
    print("DOWNLOADING ATtRACT DATABASE")
    print("=" * 70)
    print()

    attract_dir = output_dir / "attract"
    attract_dir.mkdir(parents=True, exist_ok=True)

    db_file = attract_dir / "ATtRACT.zip"
    db_txt_file ='ATtRACT_db.txt'

    # Download the database
    if not db_file.exists():
        print(f"Downloading from {ATTRACT_URL}...")
        try:
            response = requests.get(ATTRACT_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(db_file, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                    disable=total_size == 0,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"✓ Downloaded: {db_file}")

        except requests.RequestException as e:
            print(f"❌ Failed to download ATtRACT database: {e}")
            return {}

    # Parse the database
    print("\nParsing ATtRACT database...")
    rbp_motifs: Dict[str, List[Dict]] = {}
    all_motifs: List[Dict] = []

    try:
        with zipfile.ZipFile(db_file, 'r') as zip_file:
            path = zipfile.Path(zip_file, at=db_txt_file)
            # Open the file in text mode with tab delimiter
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")

                # Skip header
                next(reader, None)

                for parts in reader:
                    gene_name = parts[0].upper()
                    gene_id = parts[1] if len(parts) > 1 else ""
                    organism = parts[3] if len(parts) > 3 else ""
                    motif = parts[4] if len(parts) > 4 else ""
                    pubmed_id = parts[8] if len(parts) > 8 else ""

                    print(motif)
                    # Filter for valid motifs
                    if not motif or len(motif) < 3:
                        continue

                    motif_data = {
                        "gene_name": gene_name,
                        "gene_id": gene_id,
                        "organism": organism,
                        "motif": motif,
                        "pubmed_id": pubmed_id,
                    }

                    all_motifs.append(motif_data)

                    if gene_name not in rbp_motifs:
                        rbp_motifs[gene_name] = []
                    rbp_motifs[gene_name].append(motif_data)

        print(f"✓ Loaded {len(all_motifs):,} motifs for {len(rbp_motifs):,} RBPs")

    except Exception as e:
        print(f"❌ Error parsing ATtRACT database: {e}")
        return {}

    # Save parsed data
    motifs_file = attract_dir / "rbp_motifs.json"
    with open(motifs_file, "w") as f:
        json.dump(rbp_motifs, f, indent=2)
    print(f"✓ Saved: {motifs_file}")

    # Save human-only motifs
    human_motifs = {
        gene: [m for m in motifs if "Homo" in m.get("organism", "")]
        for gene, motifs in rbp_motifs.items()
    }
    human_motifs = {k: v for k, v in human_motifs.items() if v}

    human_file = attract_dir / "human_rbp_motifs.json"
    with open(human_file, "w") as f:
        json.dump(human_motifs, f, indent=2)
    print(f"✓ Saved: {human_file} ({len(human_motifs)} RBPs)")

    # Save motif summary
    summary_file = attract_dir / "motif_summary.tsv"
    with open(summary_file, "w") as f:
        f.write("gene_name\torganism\tmotif\tpubmed_id\n")
        for motif_data in all_motifs:
            f.write(
                f"{motif_data['gene_name']}\t{motif_data['organism']}\t"
                f"{motif_data['motif']}\t{motif_data['pubmed_id']}\n"
            )
    print(f"✓ Saved: {summary_file}")

    return rbp_motifs


# =============================================================================
# SEQUENCE GENERATION
# =============================================================================


def generate_motif_sequences(
    rbp_motifs: Dict[str, List[Dict]],
    sequences_per_motif: int = 50,
    organism_filter: Optional[str] = "Homo",
) -> List[Dict]:
    """
    Generate RNA sequences containing binding motifs

    Args:
        rbp_motifs: Dictionary of RBP motifs from ATtRACT
        sequences_per_motif: Number of sequences to generate per motif
        organism_filter: Filter by organism (None for all)

    Returns:
        List of sequence dictionaries with RBP associations
    """
    print("\nGenerating RNA sequences with binding motifs...")

    sequences = []

    for gene_name, motifs in tqdm(rbp_motifs.items(), desc="Generating sequences"):
        # Filter by organism if specified
        if organism_filter:
            motifs = [m for m in motifs if organism_filter in m.get("organism", "")]

        if not motifs:
            continue

        # Use unique motifs only
        unique_motifs = list(set(m["motif"] for m in motifs))

        for motif in unique_motifs[:5]:  # Max 5 motifs per RBP
            for _ in range(sequences_per_motif):
                rna_seq = generate_rna_with_motif(motif)
                sequences.append(
                    {
                        "sequence": rna_seq,
                        "length": len(rna_seq),
                        "rbp": gene_name,
                        "motif": motif,
                        "source": "attract_generated",
                    }
                )

    print(f"✓ Generated {len(sequences):,} sequences")
    return sequences


def generate_common_motif_sequences(
    sequences_per_rbp: int = 100,
) -> List[Dict]:
    """
    Generate RNA sequences using common literature-curated motifs

    Args:
        sequences_per_rbp: Number of sequences per RBP

    Returns:
        List of sequence dictionaries
    """
    print("\nGenerating sequences from common RBP motifs...")

    sequences = []

    for rbp_name, motifs in tqdm(COMMON_RBP_MOTIFS.items(), desc="Generating"):
        seqs_per_motif = sequences_per_rbp // len(motifs)

        for motif in motifs:
            for _ in range(seqs_per_motif):
                rna_seq = generate_rna_with_motif(motif)
                sequences.append(
                    {
                        "sequence": rna_seq,
                        "length": len(rna_seq),
                        "rbp": rbp_name,
                        "motif": motif,
                        "source": "common_motifs",
                    }
                )

    print(f"✓ Generated {len(sequences):,} sequences")
    return sequences


def generate_random_sequences(
    num_sequences: int = 10000,
    min_length: int = 30,
    max_length: int = 100,
) -> List[Dict]:
    """
    Generate random RNA sequences (for negative sampling)

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of sequence dictionaries
    """
    print(f"\nGenerating {num_sequences:,} random RNA sequences...")

    sequences = []

    for i in tqdm(range(num_sequences), desc="Generating random sequences"):
        length = random.randint(min_length, max_length)
        rna_seq = generate_random_rna(length)
        sequences.append(
            {
                "sequence": rna_seq,
                "length": length,
                "rbp": None,
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
    Load RNA sequences from FASTA file

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
                    seq = "".join(current_seq).upper().replace("T", "U")
                    if is_valid_rna(seq):
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
            seq = "".join(current_seq).upper().replace("T", "U")
            if is_valid_rna(seq):
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
    rbp_motifs: Optional[Dict] = None,
) -> None:
    """
    Save RNA sequence data

    Args:
        sequences: List of sequence dictionaries
        output_dir: Output directory
        rbp_motifs: Optional RBP motif data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as FASTA
    fasta_file = output_dir / "rna_sequences.fasta"
    with open(fasta_file, "w") as f:
        for i, seq_data in enumerate(sequences):
            seq_id = seq_data.get("seq_id", f"rna_{i}")
            rbp = seq_data.get("rbp", "")
            motif = seq_data.get("motif", "")

            header = f">{seq_id}"
            if rbp:
                header += f"|rbp={rbp}"
            if motif:
                header += f"|motif={motif}"

            f.write(f"{header}\n{seq_data['sequence']}\n")

    print(f"✓ Saved FASTA: {fasta_file}")

    # Save as JSON
    json_file = output_dir / "rna_sequences.json"
    with open(json_file, "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"✓ Saved JSON: {json_file}")

    # Save as TSV
    tsv_file = output_dir / "rna_sequences.tsv"
    with open(tsv_file, "w") as f:
        f.write("seq_id\tsequence\tlength\trbp\tmotif\tsource\n")
        for i, seq_data in enumerate(sequences):
            seq_id = seq_data.get("seq_id", f"rna_{i}")
            f.write(
                f"{seq_id}\t{seq_data['sequence']}\t{seq_data['length']}\t"
                f"{seq_data.get('rbp', '')}\t{seq_data.get('motif', '')}\t"
                f"{seq_data.get('source', '')}\n"
            )
    print(f"✓ Saved TSV: {tsv_file}")

    # Save RBP-specific files if motifs provided
    if rbp_motifs:
        rbp_dir = output_dir / "by_rbp"
        rbp_dir.mkdir(exist_ok=True)

        # Group sequences by RBP
        rbp_sequences: Dict[str, List[Dict]] = {}
        for seq_data in sequences:
            rbp = seq_data.get("rbp")
            if rbp:
                if rbp not in rbp_sequences:
                    rbp_sequences[rbp] = []
                rbp_sequences[rbp].append(seq_data)

        # Save per-RBP files
        for rbp, seqs in rbp_sequences.items():
            rbp_file = rbp_dir / f"{rbp}_sequences.fasta"
            with open(rbp_file, "w") as f:
                for i, seq_data in enumerate(seqs):
                    f.write(f">{rbp}_{i}|motif={seq_data.get('motif', '')}\n")
                    f.write(f"{seq_data['sequence']}\n")

        print(f"✓ Saved {len(rbp_sequences)} RBP-specific files to {rbp_dir}")


# =============================================================================
# METADATA AND SUMMARY
# =============================================================================


def save_metadata(
    output_dir: Path,
    source: str,
    sequence_count: int,
    rbp_count: int = 0,
    motif_count: int = 0,
) -> None:
    """Save download metadata"""
    output_file = output_dir / "download_info.json"

    info = {
        "source": source,
        "sequence_count": sequence_count,
        "rbp_count": rbp_count,
        "motif_count": motif_count,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved metadata: {output_file}")


def print_summary(
    output_dir: Path,
    sequences: List[Dict],
    rbp_motifs: Optional[Dict] = None,
) -> None:
    """Print download summary"""
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Sequences generated: {len(sequences):,}")

    if rbp_motifs:
        print(f"RBPs with motifs: {len(rbp_motifs):,}")
        total_motifs = sum(len(v) for v in rbp_motifs.values())
        print(f"Total motifs: {total_motifs:,}")

    # Count by source
    sources = {}
    for seq in sequences:
        src = seq.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\nSequences by source:")
    for src, count in sorted(sources.items()):
        print(f"  • {src}: {count:,}")

    print()
    print("Files created:")
    print("  • rna_sequences.fasta - RNA sequences (FASTA)")
    print("  • rna_sequences.json - Full data (JSON)")
    print("  • rna_sequences.tsv - Summary (TSV)")
    if rbp_motifs:
        print("  • attract/ - ATtRACT database files")
        print("  • by_rbp/ - RBP-specific sequence files")
    print("  • download_info.json - Metadata")
    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download RNA Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_rna.py --source attract
    python download_rna.py --source attract --generate-sequences
    python download_rna.py --generate-random --num-sequences 10000
    python download_rna.py --fasta-file rna_sequences.fasta
    python download_rna.py --source attract --sequences-per-motif 100

Sources:
    attract - ATtRACT database (RBP binding motifs)
    common  - Common literature-curated RBP motifs
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="attract",
        choices=["attract", "common"],
        help="Data source for RBP motifs (default: attract)",
    )

    parser.add_argument(
        "--generate-sequences",
        action="store_true",
        help="Generate RNA sequences containing binding motifs",
    )

    parser.add_argument(
        "--sequences-per-motif",
        type=int,
        default=50,
        help="Number of sequences to generate per motif (default: 50)",
    )

    parser.add_argument(
        "--generate-random",
        action="store_true",
        help="Generate random RNA sequences (for negative sampling)",
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
        default=30,
        help="Minimum sequence length (default: 30)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum sequence length (default: 100)",
    )

    parser.add_argument(
        "--fasta-file",
        type=str,
        help="Load RNA sequences from FASTA file",
    )

    parser.add_argument(
        "--organism",
        type=str,
        default="Homo",
        help="Filter motifs by organism (default: 'Homo' for human)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rna",
        help="Output directory (default: data/rna)",
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
    print("RNA DATA DOWNLOAD")
    print("=" * 70)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    sequences: List[Dict] = []
    rbp_motifs: Optional[Dict] = None

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

    # Download and process ATtRACT
    elif args.source == "attract":
        rbp_motifs = download_attract_database(output_dir)

        if args.generate_sequences and rbp_motifs:
            sequences = generate_motif_sequences(
                rbp_motifs,
                sequences_per_motif=args.sequences_per_motif,
                organism_filter=args.organism,
            )

    # Use common literature motifs
    elif args.source == "common":
        sequences = generate_common_motif_sequences(
            sequences_per_rbp=args.sequences_per_motif * 5,
        )

    # Save sequences if generated
    if sequences:
        save_sequences(sequences, output_dir, rbp_motifs)

        # Save metadata
        save_metadata(
            output_dir=output_dir,
            source=args.source if not args.fasta_file else "fasta",
            sequence_count=len(sequences),
            rbp_count=len(rbp_motifs) if rbp_motifs else 0,
            motif_count=sum(len(v) for v in rbp_motifs.values()) if rbp_motifs else 0,
        )

    elif rbp_motifs:
        # Just downloaded motifs, no sequences generated
        save_metadata(
            output_dir=output_dir,
            source=args.source,
            sequence_count=0,
            rbp_count=len(rbp_motifs),
            motif_count=sum(len(v) for v in rbp_motifs.values()),
        )

    # Print summary
    print_summary(output_dir, sequences, rbp_motifs)

    print("Done!")
    print()
    print("This RNA data can be used with:")
    print("  • download_p2r.py - Protein-RNA interactions")


if __name__ == "__main__":
    main()
