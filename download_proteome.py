#!/usr/bin/env python3
"""
Download Proteome and AlphaFold Structures
===========================================

Downloads complete proteome and AlphaFold structures for any species.

Usage:
    python download_proteome.py --species "Mycoplasma genitalium"
    python download_proteome.py --species "Escherichia coli" --strain "K-12"
    python download_proteome.py --uniprot-id UP000005640  # Direct proteome ID
    python download_proteome.py --species "Homo sapiens" --threads 20  # Parallel download
"""

import argparse
import gzip
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def search_proteome(species_name: str, strain: str = None) -> dict:
    """
    Search for proteome by species name

    Args:
        species_name: Species name (e.g., "Mycoplasma genitalium")
        strain: Optional strain name

    Returns:
        Proteome information dictionary
    """
    print(f"Searching for proteome: {species_name}")
    if strain:
        print(f"  Strain: {strain}")
    print()

    # Search UniProt proteomes
    search_query = species_name
    if strain:
        search_query += f" {strain}"

    url = "https://rest.uniprot.org/proteomes/search"
    params = {"query": search_query, "format": "json", "size": 10}

    response = requests.get(url, params=params)
    sys.stdout.flush()

    if response.status_code != 200:
        print(f"❌ Error searching proteomes: {response.status_code}")
        sys.stdout.flush()
        return None

    results = response.json()

    if not results.get("results"):
        print(f"❌ No proteomes found for: {species_name}")
        sys.stdout.flush()
        return None

    print(f"Found {len(results['results'])} proteome(s):")
    print()
    sys.stdout.flush()

    for i, proteome in enumerate(results["results"], 1):
        upid = proteome.get("id", "Unknown")
        organism = proteome.get("taxonomy", {}).get("scientificName", "Unknown")
        protein_count = proteome.get("proteinCount", 0)
        is_reference = proteome.get("isReferenceProteome", False)

        ref_marker = " [REFERENCE]" if is_reference else ""

        print(f"{i}. {upid} - {organism}{ref_marker}")
        print(f"   Proteins: {protein_count}")
        print()

    # Return first (usually best match)
    return results["results"][0]


def download_proteome_fasta(proteome_id: str, output_dir: Path) -> Path:
    """
    Download proteome FASTA file from UniProt

    Args:
        proteome_id: UniProt proteome ID (e.g., UP000000807)
        output_dir: Output directory

    Returns:
        Path to downloaded FASTA file
    """
    print(f"Downloading proteome: {proteome_id}")
    sys.stdout.flush()

    # UniProt proteome download URL
    url = f"https://rest.uniprot.org/uniprotkb/stream"
    params = {"format": "fasta", "query": f"(proteome:{proteome_id})"}

    output_file = output_dir / "proteins.fasta"
    sequences_dir = output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, params=params, stream=True)

    if response.status_code != 200:
        print(f"❌ Error downloading proteome: {response.status_code}")
        return None

    # Get total size if available
    total_size = int(response.headers.get("content-length", 0))
    downloaded_size = 0

    print("Downloading FASTA file...")
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                    progress = (downloaded_size / total_size) * 100
                    print(
                        f"  Progress: {downloaded_size / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({progress:.1f}%)"
                    )
                    sys.stdout.flush()

    # Count proteins and save individual FASTA files
    print("Splitting into individual protein files...")
    protein_count = 0
    current_id = None
    current_seq = []

    with open(output_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Save previous protein
                if current_id:
                    individual_file = sequences_dir / f"{current_id}.fasta"
                    with open(individual_file, "w") as out:
                        out.write(f">{current_id}\n{''.join(current_seq)}\n")

                # Parse new protein ID
                parts = line[1:].split("|")
                if len(parts) >= 2:
                    current_id = parts[1]
                else:
                    current_id = line[1:].split()[0]

                current_seq = []
                protein_count += 1
            else:
                current_seq.append(line.strip())

        # Save last protein
        if current_id:
            individual_file = sequences_dir / f"{current_id}.fasta"
            with open(individual_file, "w") as out:
                out.write(f">{current_id}\n{''.join(current_seq)}\n")

    print(f"✓ Downloaded {protein_count} proteins")
    print(f"  Combined FASTA: {output_file}")
    print(f"  Individual files: {sequences_dir}")
    print()

    return output_file


def download_alphafold_structure(uniprot_id: str, output_dir: Path) -> tuple:
    """
    Download AlphaFold structure for a protein

    Args:
        uniprot_id: UniProt accession (e.g., P47340)
        output_dir: Output directory

    Returns:
        Tuple of (status, message) where status is 'downloaded', 'skipped', or 'not_found'
    """
    # AlphaFold DB URL pattern (v6 is latest)
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"

    output_file = output_dir / f"{uniprot_id}.pdb"

    # Check if already exists
    if output_file.exists():
        return ("skipped", uniprot_id)

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            return ("downloaded", uniprot_id)
        else:
            return ("not_found", uniprot_id)

    except Exception:
        return ("not_found", uniprot_id)
        return (False, False)  # Failed, not skipped


def download_all_structures(fasta_file: Path, output_dir: Path, max_workers: int = 10):
    """
    Download AlphaFold structures for all proteins in FASTA (parallel)

    Args:
        fasta_file: Path to FASTA file
        output_dir: Output directory for structures
        max_workers: Number of parallel download threads (default: 10)
    """
    print("Extracting UniProt IDs from FASTA...")
    sys.stdout.flush()

    uniprot_ids = []
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Parse UniProt ID from header
                # Format: >sp|P47340|DNAB_MYCGE or >tr|Q9ZB73|...
                parts = line.split("|")
                if len(parts) >= 2:
                    uniprot_id = parts[1]
                    uniprot_ids.append(uniprot_id)

    print(f"✓ Found {len(uniprot_ids):,} protein IDs")
    print()
    sys.stdout.flush()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING ALPHAFOLD STRUCTURES")
    print("=" * 70)
    print(f"Total proteins: {len(uniprot_ids):,}")
    print(f"Output: {output_dir}")
    print(f"Parallel threads: {max_workers}")
    print()
    print("Starting download... (this may take a while for large proteomes)")
    sys.stdout.flush()

    downloaded = 0
    skipped = 0
    not_found = 0
    not_found_ids = []

    # Parallel downloads using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_id = {
            executor.submit(
                download_alphafold_structure, uniprot_id, output_dir
            ): uniprot_id
            for uniprot_id in uniprot_ids
        }

        # Process completed downloads with simple progress
        for future in as_completed(future_to_id):
            status, uniprot_id = future.result()

            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:  # not_found
                not_found += 1
                not_found_ids.append(uniprot_id)

            # Print progress every 100 proteins
            total_processed = downloaded + skipped + not_found
            if total_processed % 100 == 0:
                print(
                    f"Progress: {total_processed:,}/{len(uniprot_ids):,} | ✓ {downloaded:,} | ⊙ {skipped:,} | ✗ {not_found:,}"
                )
                sys.stdout.flush()

    print()
    print("=" * 70)
    print("ALPHAFOLD STRUCTURE DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✓ Downloaded: {downloaded:,} ({downloaded / len(uniprot_ids) * 100:.1f}%)")
    print(
        f"⊙ Skipped (already exist): {skipped:,} ({skipped / len(uniprot_ids) * 100:.1f}%)"
    )
    print(f"✗ Not found: {not_found:,} ({not_found / len(uniprot_ids) * 100:.1f}%)")
    print(f"  Total proteins: {len(uniprot_ids):,}")
    print()
    sys.stdout.flush()

    if not_found > 0 and not_found <= 20:
        print(f"Proteins not found in AlphaFold DB:")
        for uniprot_id in not_found_ids:
            print(f"  • {uniprot_id}")
        print()
    elif not_found > 20:
        print(f"First 20 proteins not found in AlphaFold DB:")
        for uniprot_id in not_found_ids[:20]:
            print(f"  • {uniprot_id}")
        print(f"  ... and {not_found - 20:,} more")
        print()

    if downloaded == 0:
        print("⚠️  WARNING: No structures were downloaded!")
        print("   This may happen if:")
        print("   • Proteins are not in AlphaFold Database")
        print("   • Network connectivity issues")
        print("   • UniProt IDs are in wrong format")
        print()


def save_proteome_info(proteome_data: dict, output_dir: Path):
    """
    Save proteome metadata

    Args:
        proteome_data: Proteome information from UniProt
        output_dir: Output directory
    """
    import json

    output_file = output_dir / "proteome_info.json"

    info = {
        "proteome_id": proteome_data.get("id"),
        "organism": proteome_data.get("taxonomy", {}).get("scientificName"),
        "taxon_id": proteome_data.get("taxonomy", {}).get("taxonId"),
        "protein_count": proteome_data.get("proteinCount"),
        "is_reference": proteome_data.get("isReferenceProteome", False),
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved proteome info: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download proteome and AlphaFold structures"
    )

    parser.add_argument(
        "--species", help="Species name (e.g., 'Mycoplasma genitalium')"
    )

    parser.add_argument("--strain", help="Strain name (optional)")

    parser.add_argument(
        "--uniprot-id", help="Direct UniProt proteome ID (e.g., UP000000807)"
    )

    parser.add_argument(
        "--output", default="data", help="Output directory (default: data)"
    )

    parser.add_argument(
        "--skip-structures",
        action="store_true",
        help="Skip downloading AlphaFold structures",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of parallel download threads (default: 10)",
    )

    args = parser.parse_args()

    if not args.species and not args.uniprot_id:
        print("❌ Error: Please provide --species or --uniprot-id")
        print()
        print("Examples:")
        print("  python download_proteome.py --species 'Mycoplasma genitalium'")
        print("  python download_proteome.py --uniprot-id UP000000807")
        return

    print("=" * 70)
    print("PROTEOME DOWNLOAD")
    print("=" * 70)
    print()
    sys.stdout.flush()

    # Search or use provided ID
    if args.uniprot_id:
        proteome_id = args.uniprot_id
        proteome_data = {"id": proteome_id}
    else:
        proteome_data = search_proteome(args.species, args.strain)
        if not proteome_data:
            return
        proteome_id = proteome_data.get("id")

    # Create output directory
    # Use species name or proteome ID as subdirectory
    if args.species:
        species_dir = args.species.replace(" ", "_").lower()
    else:
        species_dir = proteome_id.lower()

    output_dir = Path(args.output) / species_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Download FASTA
    fasta_file = download_proteome_fasta(proteome_id, output_dir)
    if not fasta_file:
        return

    # Save metadata
    if proteome_data:
        save_proteome_info(proteome_data, output_dir)
        print()

    # Download structures
    if not args.skip_structures:
        structures_dir = output_dir / "structures"
        download_all_structures(fasta_file, structures_dir, max_workers=args.threads)

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()
    print(f"✓ Proteome downloaded: {output_dir}")
    print()
    print("Files:")
    print(f"  • proteins.fasta - Combined protein sequences")
    print(f"  • sequences/ - Individual protein FASTA files")
    if not args.skip_structures:
        print(f"  • structures/ - AlphaFold structures")
    print(f"  • proteome_info.json - Metadata")
    print()


if __name__ == "__main__":
    main()
