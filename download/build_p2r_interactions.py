#!/usr/bin/env python3
"""
Build Protein-RNA Interactions
==============================

This script creates protein-RNA interaction data by matching:
1. RNA-binding proteins (RBPs) from ATtRACT with their UniProt sequences
2. RNA motifs that each RBP binds to

This creates REAL interaction pairs based on experimentally validated
binding motifs from the ATtRACT database.

Usage:
    python -m download.build_p2r_interactions
    python -m download.build_p2r_interactions --protein-dir data/proteins/homo_sapiens
    python -m download.build_p2r_interactions --rna-dir data/rna --output-dir data/rna

Output:
    protein_rna_interactions.tsv - TSV file with protein-RNA interaction pairs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def load_proteins(protein_dir: Path) -> Dict[str, Dict]:
    """
    Load protein data from UniProt download

    Args:
        protein_dir: Directory containing protein data

    Returns:
        Dictionary mapping gene names to protein info
    """
    print("Loading protein data...")

    proteins_by_gene: Dict[str, Dict] = {}
    proteins_by_uniprot: Dict[str, Dict] = {}

    # Load from proteins.json
    json_file = protein_dir / "proteins.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)

        for uniprot_id, info in data.items():
            gene_name = info.get("gene_name", "").upper()
            sequence = info.get("sequence", "")

            if gene_name and sequence:
                proteins_by_gene[gene_name] = {
                    "uniprot_id": uniprot_id,
                    "gene_name": gene_name,
                    "sequence": sequence,
                    "length": len(sequence),
                }
                proteins_by_uniprot[uniprot_id] = proteins_by_gene[gene_name]

        print(f"  ✓ Loaded {len(proteins_by_gene):,} proteins by gene name")

    # Also try loading from FASTA
    fasta_file = protein_dir / "proteins.fasta"
    if fasta_file.exists() and not proteins_by_gene:
        current_id = None
        current_seq = []

        with open(fasta_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        seq = "".join(current_seq)
                        proteins_by_uniprot[current_id] = {
                            "uniprot_id": current_id,
                            "sequence": seq,
                            "length": len(seq),
                        }

                    # Parse header: >sp|P12345|GENE_HUMAN ...
                    parts = line[1:].split("|")
                    if len(parts) >= 2:
                        current_id = parts[1]
                    else:
                        current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Last sequence
            if current_id and current_seq:
                seq = "".join(current_seq)
                proteins_by_uniprot[current_id] = {
                    "uniprot_id": current_id,
                    "sequence": seq,
                    "length": len(seq),
                }

        print(f"  ✓ Loaded {len(proteins_by_uniprot):,} proteins from FASTA")

    return proteins_by_gene, proteins_by_uniprot


def load_rbp_motifs(rna_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load RBP motifs from ATtRACT

    Args:
        rna_dir: Directory containing RNA data

    Returns:
        Dictionary mapping RBP gene names to their binding motifs
    """
    print("Loading RBP motifs from ATtRACT...")

    rbp_motifs: Dict[str, List[Dict]] = {}

    # Try human-specific motifs first
    human_file = rna_dir / "attract" / "human_rbp_motifs.json"
    if human_file.exists():
        with open(human_file) as f:
            rbp_motifs = json.load(f)
        print(f"  ✓ Loaded motifs for {len(rbp_motifs):,} human RBPs")
        return rbp_motifs

    # Fall back to all motifs
    all_file = rna_dir / "attract" / "rbp_motifs.json"
    if all_file.exists():
        with open(all_file) as f:
            all_motifs = json.load(f)

        # Filter for human
        for gene, motifs in all_motifs.items():
            human_motifs = [m for m in motifs if "Homo" in m.get("organism", "")]
            if human_motifs:
                rbp_motifs[gene.upper()] = human_motifs

        print(f"  ✓ Loaded motifs for {len(rbp_motifs):,} human RBPs")

    return rbp_motifs


def load_rna_sequences(rna_dir: Path) -> Dict[str, str]:
    """
    Load RNA sequences from Ensembl/RNAcentral downloads

    Args:
        rna_dir: Directory containing RNA data

    Returns:
        Dictionary mapping RNA IDs to sequences
    """
    print("Loading RNA sequences...")

    rna_sequences: Dict[str, str] = {}

    # Try JSON file first
    json_file = rna_dir / "rna_sequences.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)

        for entry in data:
            rna_id = entry.get("id", "")
            sequence = entry.get("sequence", "")
            if rna_id and sequence:
                rna_sequences[rna_id] = sequence

        print(f"  ✓ Loaded {len(rna_sequences):,} RNA sequences")

    # Also check Ensembl directory
    ensembl_json = rna_dir / "ensembl" / "ncrna_sequences.json"
    if ensembl_json.exists():
        with open(ensembl_json) as f:
            data = json.load(f)

        for entry in data:
            rna_id = entry.get("id", "")
            sequence = entry.get("sequence", "")
            if rna_id and sequence:
                rna_sequences[rna_id] = sequence

        print(f"  ✓ Total RNA sequences: {len(rna_sequences):,}")

    return rna_sequences


def build_interactions(
    proteins_by_gene: Dict[str, Dict],
    proteins_by_uniprot: Dict[str, Dict],
    rbp_motifs: Dict[str, List[Dict]],
    rna_sequences: Dict[str, str],
    min_motif_length: int = 5,
    max_motif_length: int = 50,
) -> List[Dict]:
    """
    Build protein-RNA interaction pairs

    Args:
        proteins_by_gene: Proteins indexed by gene name
        proteins_by_uniprot: Proteins indexed by UniProt ID
        rbp_motifs: RBP binding motifs from ATtRACT
        rna_sequences: RNA sequences (optional, for extended sequences)
        min_motif_length: Minimum motif length to include
        max_motif_length: Maximum motif length to include

    Returns:
        List of interaction dictionaries
    """
    print("\nBuilding protein-RNA interactions...")

    interactions = []
    matched_rbps = set()
    unmatched_rbps = set()

    for rbp_name, motifs in rbp_motifs.items():
        rbp_upper = rbp_name.upper()

        # Try to find protein sequence for this RBP
        protein_info = None

        # Try direct gene name match
        if rbp_upper in proteins_by_gene:
            protein_info = proteins_by_gene[rbp_upper]

        # Try common aliases
        if not protein_info:
            aliases = {
                "HNRNPA1": ["HNRPA1"],
                "HNRNPC": ["HNRPC"],
                "ELAVL1": ["HUR", "HUA"],
                "TARDBP": ["TDP43", "TDP-43"],
                "FUS": ["TLS"],
                "PTBP1": ["PTB"],
                "RBFOX2": ["FOX2", "RBM9"],
                "QKI": ["QK"],
                "IGF2BP1": ["IMP1", "ZBP1"],
                "IGF2BP2": ["IMP2"],
                "IGF2BP3": ["IMP3"],
                "KHDRBS1": ["SAM68"],
                "CELF1": ["CUGBP1", "BRUNOL2"],
                "MBNL1": ["EXP"],
                "YTHDF1": ["YTHDF1"],
                "YTHDF2": ["YTHDF2"],
                "YTHDC1": ["YTHDC1"],
            }

            for alias in aliases.get(rbp_upper, []):
                if alias in proteins_by_gene:
                    protein_info = proteins_by_gene[alias]
                    break

        if not protein_info:
            unmatched_rbps.add(rbp_name)
            continue

        matched_rbps.add(rbp_name)
        protein_seq = protein_info["sequence"]
        uniprot_id = protein_info["uniprot_id"]

        # Get unique motifs for this RBP
        seen_motifs = set()

        for motif_data in motifs:
            motif = motif_data.get("motif", "")

            # Clean and validate motif
            motif = motif.upper().replace("T", "U")

            # Skip if too short or too long
            if len(motif) < min_motif_length or len(motif) > max_motif_length:
                continue

            # Skip if contains invalid characters
            valid_chars = set("ACGUN")
            if not all(c in valid_chars for c in motif):
                continue

            # Skip duplicates
            if motif in seen_motifs:
                continue
            seen_motifs.add(motif)

            # Create interaction
            interaction = {
                "protein_id": uniprot_id,
                "protein_seq": protein_seq,
                "rna_seq": motif,
                "label": 1,
                "source": "ATtRACT",
                "rbp_name": rbp_name,
                "pubmed_id": motif_data.get("pubmed_id", ""),
            }
            interactions.append(interaction)

    print(f"  ✓ Matched {len(matched_rbps):,} RBPs to UniProt sequences")
    print(f"  ✓ Could not match {len(unmatched_rbps):,} RBPs")
    print(f"  ✓ Created {len(interactions):,} protein-RNA interactions")

    if unmatched_rbps and len(unmatched_rbps) <= 20:
        print(f"  Unmatched RBPs: {', '.join(sorted(unmatched_rbps))}")

    return interactions


def save_interactions(
    interactions: List[Dict],
    output_dir: Path,
) -> Path:
    """
    Save interactions to TSV file

    Args:
        interactions: List of interaction dictionaries
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    print("\nSaving interactions...")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "protein_rna_interactions.tsv"

    with open(output_file, "w") as f:
        # Header
        f.write(
            "protein_id\tprotein_seq\trna_seq\tlabel\tsource\trbp_name\tpubmed_id\n"
        )

        for interaction in interactions:
            f.write(
                f"{interaction['protein_id']}\t"
                f"{interaction['protein_seq']}\t"
                f"{interaction['rna_seq']}\t"
                f"{interaction['label']}\t"
                f"{interaction['source']}\t"
                f"{interaction['rbp_name']}\t"
                f"{interaction['pubmed_id']}\n"
            )

    print(f"✓ Saved: {output_file}")

    # Also save as JSON for easier loading
    json_file = output_dir / "protein_rna_interactions.json"
    with open(json_file, "w") as f:
        json.dump(interactions, f, indent=2)
    print(f"✓ Saved: {json_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("INTERACTION SUMMARY")
    print("=" * 60)

    unique_proteins = len(set(i["protein_id"] for i in interactions))
    unique_rna = len(set(i["rna_seq"] for i in interactions))
    unique_rbps = len(set(i["rbp_name"] for i in interactions))

    print(f"Total interactions: {len(interactions):,}")
    print(f"Unique proteins: {unique_proteins:,}")
    print(f"Unique RNA motifs: {unique_rna:,}")
    print(f"Unique RBPs: {unique_rbps:,}")

    # RNA length distribution
    rna_lengths = [len(i["rna_seq"]) for i in interactions]
    print(f"RNA length range: {min(rna_lengths)}-{max(rna_lengths)}")
    print(f"Mean RNA length: {sum(rna_lengths) / len(rna_lengths):.1f}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Build Protein-RNA Interactions from ATtRACT Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m download.build_p2r_interactions
    python -m download.build_p2r_interactions --protein-dir data/proteins/homo_sapiens
    python -m download.build_p2r_interactions --min-motif-length 6 --max-motif-length 30
        """,
    )

    parser.add_argument(
        "--protein-dir",
        type=str,
        default="data/proteins/homo_sapiens",
        help="Directory containing protein data (default: data/proteins/homo_sapiens)",
    )

    parser.add_argument(
        "--rna-dir",
        type=str,
        default="data/rna",
        help="Directory containing RNA data (default: data/rna)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rna",
        help="Output directory (default: data/rna)",
    )

    parser.add_argument(
        "--min-motif-length",
        type=int,
        default=5,
        help="Minimum motif length (default: 5)",
    )

    parser.add_argument(
        "--max-motif-length",
        type=int,
        default=50,
        help="Maximum motif length (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BUILD PROTEIN-RNA INTERACTIONS")
    print("=" * 70)
    print()
    print("This script builds protein-RNA interaction pairs from:")
    print("  • UniProt protein sequences")
    print("  • ATtRACT experimentally validated RBP binding motifs")
    print()

    protein_dir = Path(args.protein_dir)
    rna_dir = Path(args.rna_dir)
    output_dir = Path(args.output_dir)

    # Check directories exist
    if not protein_dir.exists():
        print(f"❌ Protein directory not found: {protein_dir}")
        print("   Run: python -m download.download_pro --species 'Homo sapiens'")
        sys.exit(1)

    if not rna_dir.exists():
        print(f"❌ RNA directory not found: {rna_dir}")
        print("   Run: python -m download.download_rna --source attract")
        sys.exit(1)

    # Load data
    proteins_by_gene, proteins_by_uniprot = load_proteins(protein_dir)
    rbp_motifs = load_rbp_motifs(rna_dir)
    rna_sequences = load_rna_sequences(rna_dir)

    if not proteins_by_gene and not proteins_by_uniprot:
        print("❌ No proteins loaded!")
        sys.exit(1)

    if not rbp_motifs:
        print("❌ No RBP motifs loaded!")
        print("   Run: python -m download.download_rna --source attract")
        sys.exit(1)

    # Build interactions
    interactions = build_interactions(
        proteins_by_gene,
        proteins_by_uniprot,
        rbp_motifs,
        rna_sequences,
        min_motif_length=args.min_motif_length,
        max_motif_length=args.max_motif_length,
    )

    if not interactions:
        print("❌ No interactions could be built!")
        print("   Check that protein gene names match RBP names from ATtRACT")
        sys.exit(1)

    # Save interactions
    save_interactions(interactions, output_dir)

    print("\nDone!")
    print()
    print("Next steps:")
    print("  python -m prepare.prepare_p2r_data")
    print()


if __name__ == "__main__":
    main()
