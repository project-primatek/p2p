#!/usr/bin/env python3
"""
Download STRING Database Interactions
======================================

Downloads protein-protein interaction data from STRING database for any species.

Usage:
    python download_string.py --species "Mycoplasma genitalium"
    python download_string.py --taxon-id 243273
    python download_string.py --taxon-id 511145 --score 700  # E. coli, high confidence
"""

import argparse
import gzip
import time
from pathlib import Path

import requests
from tqdm import tqdm


def search_species(species_name: str) -> dict:
    """
    Search for species in STRING database

    Args:
        species_name: Species name (e.g., "Mycoplasma genitalium")

    Returns:
        Species information dictionary
    """
    print(f"Searching STRING database for: {species_name}")
    print()

    # STRING API endpoint for species search
    url = "https://string-db.org/api/json/resolve"
    params = {"identifier": species_name, "species": ""}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            print(f"❌ Error searching STRING: {response.status_code}")
            return None

        results = response.json()

        if not results:
            print(f"❌ No species found for: {species_name}")
            return None

        # Group by taxon ID
        species_map = {}
        for item in results:
            taxon_id = item.get("taxonId")
            if taxon_id not in species_map:
                species_map[taxon_id] = {
                    "taxon_id": taxon_id,
                    "species_name": item.get("taxonName", "Unknown"),
                    "count": 0,
                }
            species_map[taxon_id]["count"] += 1

        print(f"Found {len(species_map)} species match(es):")
        print()

        for i, (taxon_id, info) in enumerate(species_map.items(), 1):
            print(f"{i}. {info['species_name']}")
            print(f"   Taxon ID: {taxon_id}")
            print(f"   Proteins: {info['count']}")
            print()

        # Return first match
        first_taxon = list(species_map.values())[0]
        return first_taxon

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_taxon_info(taxon_id: int) -> dict:
    """
    Get information about a taxon from STRING

    Args:
        taxon_id: NCBI Taxonomy ID

    Returns:
        Taxon information
    """
    print(f"Getting info for taxon ID: {taxon_id}")
    print()

    # STRING doesn't have a direct taxon info endpoint
    # We'll validate by trying to get network stats
    url = "https://string-db.org/api/json/network_stats"
    params = {"identifiers": f"{taxon_id}.dummy"}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return {
                "taxon_id": taxon_id,
                "species_name": f"Taxon {taxon_id}",
                "found": True,
            }
        else:
            return {"taxon_id": taxon_id, "found": False}
    except:
        return {"taxon_id": taxon_id, "found": False}


def download_string_interactions(
    taxon_id: int, output_dir: Path, score_threshold: int = 400
) -> Path:
    """
    Download STRING protein-protein interactions

    Args:
        taxon_id: NCBI Taxonomy ID
        output_dir: Output directory
        score_threshold: Minimum confidence score (0-1000, default 400 = medium confidence)

    Returns:
        Path to downloaded file
    """
    print(f"Downloading STRING interactions for taxon: {taxon_id}")
    print(f"  Score threshold: {score_threshold} (0-1000 scale)")
    print()

    # STRING download URL pattern
    # Format: [taxon_id].protein.links.v12.0.txt.gz
    version = "12.0"
    filename = f"{taxon_id}.protein.links.v{version}.txt.gz"
    url = f"https://stringdb-downloads.org/download/protein.links.v{version}/{filename}"

    output_file = output_dir / filename
    output_file_unzipped = output_dir / f"string_interactions.tsv"

    print(f"Downloading from: {url}")
    print()

    try:
        response = requests.get(url, stream=True, timeout=30)

        if response.status_code != 200:
            print(f"❌ Error downloading STRING data: {response.status_code}")
            print(f"   This taxon may not be available in STRING")
            return None

        # Get total size
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress bar
        with (
            open(output_file, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ Downloaded: {output_file}")
        print()

        # Decompress and filter by score
        print(f"Decompressing and filtering (score >= {score_threshold})...")

        interaction_count = 0
        filtered_count = 0

        with (
            gzip.open(output_file, "rt") as f_in,
            open(output_file_unzipped, "w") as f_out,
        ):
            # Read header
            header = f_in.readline()
            f_out.write(header)

            # Process interactions
            for line in tqdm(f_in, desc="Processing"):
                interaction_count += 1
                parts = line.strip().split()

                if len(parts) >= 3:
                    score = int(parts[2])
                    if score >= score_threshold:
                        f_out.write(line)
                        filtered_count += 1

        print()
        print(f"✓ Total interactions: {interaction_count:,}")
        print(f"✓ Filtered (score >= {score_threshold}): {filtered_count:,}")
        print(f"✓ Saved to: {output_file_unzipped}")
        print()

        # Optionally remove gzipped file to save space
        # output_file.unlink()

        return output_file_unzipped

    except requests.exceptions.Timeout:
        print("❌ Download timed out. STRING servers may be slow.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def download_string_info(taxon_id: int, output_dir: Path) -> Path:
    """
    Download STRING protein info (names, annotations)

    Args:
        taxon_id: NCBI Taxonomy ID
        output_dir: Output directory

    Returns:
        Path to downloaded file
    """
    print(f"Downloading protein info for taxon: {taxon_id}")

    version = "12.0"
    filename = f"{taxon_id}.protein.info.v{version}.txt.gz"
    url = f"https://stringdb-downloads.org/download/protein.info.v{version}/{filename}"

    output_file = output_dir / filename
    output_file_unzipped = output_dir / "string_protein_info.tsv"

    try:
        response = requests.get(url, stream=True, timeout=30)

        if response.status_code != 200:
            print(f"⚠️  Warning: Could not download protein info")
            return None

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(output_file, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading info"
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Decompress
        print("Decompressing...")
        with (
            gzip.open(output_file, "rt") as f_in,
            open(output_file_unzipped, "w") as f_out,
        ):
            for line in f_in:
                f_out.write(line)

        print(f"✓ Saved to: {output_file_unzipped}")
        print()

        return output_file_unzipped

    except Exception as e:
        print(f"⚠️  Warning: {e}")
        return None


def save_download_info(taxon_id: int, species_name: str, output_dir: Path):
    """
    Save metadata about the download

    Args:
        taxon_id: NCBI Taxonomy ID
        species_name: Species name
        output_dir: Output directory
    """
    import json

    output_file = output_dir / "string_info.json"

    info = {
        "taxon_id": taxon_id,
        "species_name": species_name,
        "database": "STRING v12.0",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "https://string-db.org",
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved download info: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download STRING database interactions"
    )

    parser.add_argument(
        "--species", help="Species name (e.g., 'Mycoplasma genitalium')"
    )

    parser.add_argument("--taxon-id", type=int, help="NCBI Taxonomy ID (e.g., 243273)")

    parser.add_argument(
        "--score",
        type=int,
        default=400,
        help="Minimum confidence score 0-1000 (default: 400 = medium confidence)",
    )

    parser.add_argument(
        "--output", default="data", help="Output directory (default: data)"
    )

    parser.add_argument(
        "--skip-protein-info",
        action="store_true",
        help="Skip downloading protein annotation info",
    )

    args = parser.parse_args()

    if not args.species and not args.taxon_id:
        print("❌ Error: Please provide --species or --taxon-id")
        print()
        print("Examples:")
        print("  python download_string.py --species 'Mycoplasma genitalium'")
        print("  python download_string.py --taxon-id 243273")
        print("  python download_string.py --taxon-id 511145 --score 700")
        print()
        print("Common taxon IDs:")
        print("  243273 - Mycoplasma genitalium")
        print("  511145 - Escherichia coli K-12")
        print("  224308 - Bacillus subtilis")
        print("  83332  - Mycobacterium tuberculosis")
        return

    print("=" * 70)
    print("STRING DATABASE DOWNLOAD")
    print("=" * 70)
    print()

    # Get taxon ID
    if args.taxon_id:
        taxon_id = args.taxon_id
        species_name = f"Taxon {taxon_id}"
    else:
        species_info = search_species(args.species)
        if not species_info:
            return
        taxon_id = species_info["taxon_id"]
        species_name = species_info["species_name"]

    # Create output directory
    if args.species:
        species_dir = args.species.replace(" ", "_").lower()
    else:
        species_dir = f"taxon_{taxon_id}"

    output_dir = Path(args.output) / species_dir / "string"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Download interactions
    interactions_file = download_string_interactions(taxon_id, output_dir, args.score)

    if not interactions_file:
        print("❌ Failed to download interactions")
        return

    # Download protein info
    if not args.skip_protein_info:
        download_string_info(taxon_id, output_dir)

    # Save metadata
    save_download_info(taxon_id, species_name, output_dir)
    print()

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()
    print(f"✓ STRING data downloaded: {output_dir}")
    print()
    print("Files:")
    print(f"  • string_interactions.tsv - Protein-protein interactions")
    if not args.skip_protein_info:
        print(f"  • string_protein_info.tsv - Protein annotations")
    print(f"  • string_info.json - Metadata")
    print()
    print("Confidence scores:")
    print("  • 0-150:   Low confidence")
    print("  • 150-400: Medium confidence")
    print("  • 400-700: High confidence")
    print("  • 700-900: Very high confidence")
    print("  • 900-1000: Highest confidence")
    print()


if __name__ == "__main__":
    main()
