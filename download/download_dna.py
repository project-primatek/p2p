#!/usr/bin/env python3
"""
Download DNA Data
=================

Downloads real DNA-protein interaction data from multiple sources:
1. ENCODE ChIP-seq - Real transcription factor binding sites from experiments
2. JASPAR - Real experimentally validated TF binding motifs (no sequence generation)

This script provides REAL DNA data for protein-DNA interaction models.
NO synthetic/generated sequences are created.

Usage:
    python -m download.download_dna --source encode
    python -m download.download_dna --source jaspar
    python -m download.download_dna --source all
    python -m download.download_dna --fasta-file dna_sequences.fasta

Data Sources:
    - ENCODE ChIP-seq: Real experimentally determined TF binding sites
    - JASPAR: Experimentally validated transcription factor binding motifs
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

# JASPAR API
JASPAR_API_URL = "https://jaspar.elixir.no/api/v1"

# ENCODE API
ENCODE_API_URL = "https://www.encodeproject.org"

# =============================================================================
# DNA VOCABULARY AND UTILITIES
# =============================================================================

DNA_NUCLEOTIDES = set("ACGT")


def is_valid_dna(sequence: str) -> bool:
    """Check if sequence is valid DNA"""
    if not sequence:
        return False
    seq_upper = sequence.upper()
    return all(c in DNA_NUCLEOTIDES or c == "N" for c in seq_upper)


def reverse_complement(sequence: str) -> str:
    """Get reverse complement of DNA sequence"""
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(sequence.upper()))


# =============================================================================
# ENCODE ChIP-seq DATA DOWNLOAD
# =============================================================================


def download_encode_chipseq(
    output_dir: Path,
    organism: str = "Homo sapiens",
    max_experiments: int = 100,
    max_binding_sites: int = 100000,
) -> Dict[str, List[Dict]]:
    """
    Download real ChIP-seq data from ENCODE

    ChIP-seq (Chromatin Immunoprecipitation sequencing) provides
    experimentally determined transcription factor binding sites.

    Args:
        output_dir: Output directory
        organism: Target organism
        max_experiments: Maximum number of experiments to download
        max_binding_sites: Maximum binding sites per TF

    Returns:
        Dictionary mapping TF names to binding site data
    """
    print("=" * 70)
    print("DOWNLOADING ENCODE ChIP-seq DATA")
    print("=" * 70)
    print(f"Organism: {organism}")
    print(f"Max experiments: {max_experiments}")
    print()

    encode_dir = output_dir / "encode"
    encode_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Accept": "application/json"}

    # Search for TF ChIP-seq experiments
    print("Searching for TF ChIP-seq experiments...")

    search_url = (
        f"{ENCODE_API_URL}/search/"
        f"?type=Experiment"
        f"&assay_title=TF+ChIP-seq"
        f"&replicates.library.biosample.donor.organism.scientific_name={organism.replace(' ', '+')}"
        f"&status=released"
        f"&limit={max_experiments}"
        f"&format=json"
    )

    try:
        response = requests.get(search_url, headers=headers, timeout=60)
        response.raise_for_status()
        search_results = response.json()
    except requests.RequestException as e:
        print(f"❌ Failed to search ENCODE: {e}")
        return {}

    experiments = search_results.get("@graph", [])
    print(f"✓ Found {len(experiments)} TF ChIP-seq experiments")

    if not experiments:
        print("No experiments found. Try different search parameters.")
        return {}

    # Process each experiment
    tf_binding_sites: Dict[str, List[Dict]] = {}
    all_binding_sites: List[Dict] = []

    for exp in tqdm(experiments, desc="Processing experiments"):
        exp_accession = exp.get("accession", "")

        # Get target (TF) information
        targets = exp.get("target", [])
        if isinstance(targets, str):
            targets = [targets]

        # Get the TF name from the target
        tf_name = None
        for target in targets:
            if isinstance(target, dict):
                tf_name = target.get("label", "").upper()
            elif isinstance(target, str):
                # Target is a path like "/targets/CTCF-human/"
                parts = target.strip("/").split("/")
                if len(parts) >= 2:
                    tf_name = parts[-1].split("-")[0].upper()
            if tf_name:
                break

        if not tf_name:
            continue

        # Get experiment details for files
        exp_url = f"{ENCODE_API_URL}/experiments/{exp_accession}/?format=json"

        try:
            exp_response = requests.get(exp_url, headers=headers, timeout=30)
            exp_response.raise_for_status()
            exp_data = exp_response.json()
        except requests.RequestException:
            continue

        # Find BED narrowPeak files (binding sites)
        files = exp_data.get("files", [])

        for file_ref in files:
            if isinstance(file_ref, str):
                file_id = file_ref.strip("/").split("/")[-1]
            else:
                file_id = file_ref.get("accession", "")

            if not file_id:
                continue

            # Get file metadata
            file_url = f"{ENCODE_API_URL}/files/{file_id}/?format=json"

            try:
                file_response = requests.get(file_url, headers=headers, timeout=30)
                file_response.raise_for_status()
                file_data = file_response.json()
            except requests.RequestException:
                continue

            # Look for BED narrowPeak files (contain binding peaks)
            file_format = file_data.get("file_format", "")
            output_type = file_data.get("output_type", "")

            if file_format == "bed" and "peak" in output_type.lower():
                # Download the BED file
                download_url = file_data.get("cloud_metadata", {}).get("url")
                if not download_url:
                    download_url = f"{ENCODE_API_URL}{file_data.get('href', '')}"

                if not download_url or download_url == ENCODE_API_URL:
                    continue

                try:
                    bed_response = requests.get(download_url, timeout=60)
                    bed_response.raise_for_status()

                    # Parse BED content
                    content = bed_response.content
                    if download_url.endswith(".gz"):
                        content = gzip.decompress(content)

                    lines = content.decode("utf-8", errors="ignore").strip().split("\n")

                    sites_for_tf = []
                    for line in lines[:max_binding_sites]:
                        if line.startswith("#") or not line.strip():
                            continue

                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue

                        chrom = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        name = parts[3] if len(parts) > 3 else ""
                        score = float(parts[4]) if len(parts) > 4 else 0.0
                        strand = parts[5] if len(parts) > 5 else "+"

                        site_data = {
                            "tf_name": tf_name,
                            "chromosome": chrom,
                            "start": start,
                            "end": end,
                            "strand": strand,
                            "score": score,
                            "name": name,
                            "experiment": exp_accession,
                            "source": "ENCODE_ChIP-seq",
                        }

                        sites_for_tf.append(site_data)
                        all_binding_sites.append(site_data)

                    if sites_for_tf:
                        if tf_name not in tf_binding_sites:
                            tf_binding_sites[tf_name] = []
                        tf_binding_sites[tf_name].extend(sites_for_tf)

                    # Only process one BED file per experiment
                    break

                except requests.RequestException:
                    continue

        # Rate limiting
        time.sleep(0.1)

    print(f"\n✓ Downloaded binding sites for {len(tf_binding_sites)} TFs")
    print(f"✓ Total binding sites: {len(all_binding_sites):,}")

    # Save results
    if tf_binding_sites:
        # Save all binding sites as JSON
        sites_file = encode_dir / "binding_sites.json"
        with open(sites_file, "w") as f:
            json.dump(tf_binding_sites, f, indent=2)
        print(f"✓ Saved: {sites_file}")

        # Save as TSV
        tsv_file = encode_dir / "binding_sites.tsv"
        with open(tsv_file, "w") as f:
            f.write("tf_name\tchromosome\tstart\tend\tstrand\tscore\texperiment\n")
            for site in all_binding_sites:
                f.write(
                    f"{site['tf_name']}\t{site['chromosome']}\t{site['start']}\t"
                    f"{site['end']}\t{site['strand']}\t{site['score']}\t"
                    f"{site['experiment']}\n"
                )
        print(f"✓ Saved: {tsv_file}")

        # Save summary
        summary_file = encode_dir / "summary.json"
        summary = {
            "total_tfs": len(tf_binding_sites),
            "total_binding_sites": len(all_binding_sites),
            "tf_counts": {tf: len(sites) for tf, sites in tf_binding_sites.items()},
            "organism": organism,
            "source": "ENCODE_ChIP-seq",
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {summary_file}")

    return tf_binding_sites


# =============================================================================
# JASPAR DATABASE DOWNLOAD
# =============================================================================


def download_jaspar_motifs(
    output_dir: Path,
    collection: str = "CORE",
    tax_group: str = "vertebrates",
    tf_class: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Download transcription factor binding motifs from JASPAR

    JASPAR contains experimentally validated position frequency matrices
    for transcription factor binding sites.

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
    print(f"Collection: {collection}")
    print(f"Taxonomic group: {tax_group}")
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

    # Save PFM data
    pfm_file = jaspar_dir / "position_frequency_matrices.json"
    pfm_data = {mid: m.get("pfm", {}) for mid, m in motifs.items()}
    with open(pfm_file, "w") as f:
        json.dump(pfm_data, f, indent=2)
    print(f"✓ Saved: {pfm_file}")

    return motifs


# =============================================================================
# FILE I/O
# =============================================================================


def load_fasta_file(fasta_path: Path) -> List[Dict]:
    """
    Load DNA sequences from a FASTA file

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of sequence dictionaries
    """
    print(f"\nLoading sequences from {fasta_path}...")

    sequences = []
    current_id = None
    current_seq = []
    current_meta = {}

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence
                if current_id and current_seq:
                    seq = "".join(current_seq).upper()
                    if is_valid_dna(seq):
                        sequences.append(
                            {
                                "id": current_id,
                                "sequence": seq,
                                "length": len(seq),
                                "source": "fasta",
                                **current_meta,
                            }
                        )

                # Parse header
                header = line[1:]
                parts = header.split("|")
                current_id = parts[0].strip()
                current_meta = {}

                # Parse additional metadata from header
                if len(parts) > 1:
                    for part in parts[1:]:
                        if "=" in part:
                            key, value = part.split("=", 1)
                            current_meta[key.strip()] = value.strip()
                        else:
                            current_meta["description"] = part.strip()

                current_seq = []
            else:
                current_seq.append(line)

    # Don't forget last sequence
    if current_id and current_seq:
        seq = "".join(current_seq).upper()
        if is_valid_dna(seq):
            sequences.append(
                {
                    "id": current_id,
                    "sequence": seq,
                    "length": len(seq),
                    "source": "fasta",
                    **current_meta,
                }
            )

    print(f"✓ Loaded {len(sequences):,} valid DNA sequences")
    return sequences


def save_sequences(
    sequences: List[Dict],
    output_dir: Path,
) -> None:
    """
    Save DNA sequences to multiple formats

    Args:
        sequences: List of sequence dictionaries
        output_dir: Output directory
    """
    if not sequences:
        print("No sequences to save.")
        return

    print(f"\nSaving {len(sequences):,} sequences...")

    # Save as JSON
    json_file = output_dir / "dna_sequences.json"
    with open(json_file, "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"✓ Saved: {json_file}")

    # Save as FASTA
    fasta_file = output_dir / "dna_sequences.fasta"
    with open(fasta_file, "w") as f:
        for seq_data in sequences:
            seq_id = seq_data.get("id", "unknown")
            tf = seq_data.get("tf_name", "")
            header = f">{seq_id}"
            if tf:
                header += f"|tf={tf}"
            f.write(header + "\n")

            # Write sequence in lines of 80 characters
            seq = seq_data["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
    print(f"✓ Saved: {fasta_file}")

    # Save as TSV
    tsv_file = output_dir / "dna_sequences.tsv"
    with open(tsv_file, "w") as f:
        f.write("id\ttf_name\tlength\tsource\tsequence\n")
        for seq_data in sequences:
            f.write(
                f"{seq_data.get('id', 'unknown')}\t"
                f"{seq_data.get('tf_name', '')}\t"
                f"{seq_data.get('length', len(seq_data['sequence']))}\t"
                f"{seq_data.get('source', 'unknown')}\t"
                f"{seq_data['sequence']}\n"
            )
    print(f"✓ Saved: {tsv_file}")


def save_metadata(
    output_dir: Path,
    source: str,
    sequence_count: int = 0,
    tf_count: int = 0,
    motif_count: int = 0,
    binding_site_count: int = 0,
) -> None:
    """Save download metadata"""
    output_file = output_dir / "download_info.json"

    info = {
        "source": source,
        "sequence_count": sequence_count,
        "tf_count": tf_count,
        "motif_count": motif_count,
        "binding_site_count": binding_site_count,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved metadata: {output_file}")


def print_summary(
    output_dir: Path,
    sequences: List[Dict],
    motifs: Optional[Dict] = None,
    binding_sites: Optional[Dict] = None,
) -> None:
    """Print download summary"""
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Sequences: {len(sequences):,}")

    if motifs:
        print(f"TFs with motifs: {len(motifs):,}")

    if binding_sites:
        print(f"TFs with binding sites: {len(binding_sites):,}")
        total_sites = sum(len(v) for v in binding_sites.values())
        print(f"Total binding sites: {total_sites:,}")

    # Count by source
    if sequences:
        sources = {}
        for seq in sequences:
            src = seq.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        print("\nSequences by source:")
        for src, count in sorted(sources.items()):
            print(f"  • {src}: {count:,}")

    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download DNA Data (Real Data Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m download.download_dna --source encode
    python -m download.download_dna --source jaspar
    python -m download.download_dna --source all
    python -m download.download_dna --fasta-file dna_sequences.fasta

Sources:
    encode  - ENCODE ChIP-seq data (real TF binding sites)
    jaspar  - JASPAR database (experimentally validated TF motifs)
    all     - Download from all sources
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="jaspar",
        choices=["encode", "jaspar", "all"],
        help="Data source (default: jaspar)",
    )

    parser.add_argument(
        "--fasta-file",
        type=str,
        help="Load DNA sequences from FASTA file",
    )

    parser.add_argument(
        "--organism",
        type=str,
        default="Homo sapiens",
        help="Target organism (default: 'Homo sapiens')",
    )

    parser.add_argument(
        "--max-experiments",
        type=int,
        default=100,
        help="Maximum ENCODE experiments to process (default: 100)",
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
        "--output-dir",
        type=str,
        default="data/dna",
        help="Output directory (default: data/dna)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DNA DATA DOWNLOAD (REAL DATA ONLY)")
    print("=" * 70)
    print()
    print("NOTE: This script downloads REAL experimental data only.")
    print("      No synthetic/generated sequences are created.")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    sequences: List[Dict] = []
    motifs: Optional[Dict] = None
    binding_sites: Optional[Dict] = None

    # Load from FASTA file if specified
    if args.fasta_file:
        sequences = load_fasta_file(Path(args.fasta_file))

    elif args.source == "encode" or args.source == "all":
        # Download ENCODE ChIP-seq data
        binding_sites = download_encode_chipseq(
            output_dir,
            organism=args.organism,
            max_experiments=args.max_experiments,
        )

        if args.source == "all":
            # Also download JASPAR
            motifs = download_jaspar_motifs(
                output_dir,
                collection=args.collection,
                tax_group=args.tax_group,
            )

    elif args.source == "jaspar":
        motifs = download_jaspar_motifs(
            output_dir,
            collection=args.collection,
            tax_group=args.tax_group,
        )

    # Save sequences if we have them
    if sequences:
        save_sequences(sequences, output_dir)

    # Save metadata
    save_metadata(
        output_dir=output_dir,
        source=args.source if not args.fasta_file else "fasta",
        sequence_count=len(sequences),
        tf_count=len(motifs) if motifs else 0,
        motif_count=len(motifs) if motifs else 0,
        binding_site_count=sum(len(v) for v in binding_sites.values())
        if binding_sites
        else 0,
    )

    # Print summary
    print_summary(output_dir, sequences, motifs, binding_sites)

    print("Done!")
    print()
    print("This DNA data can be used with:")
    print("  • python -m prepare.prepare_p2d_data - Prepare protein-DNA data")
    print()


if __name__ == "__main__":
    main()
