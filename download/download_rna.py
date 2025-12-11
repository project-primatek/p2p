#!/usr/bin/env python3
"""
Download RNA Data
=================

Downloads real RNA-protein interaction data from multiple sources:
1. ENCODE eCLIP - Real RNA binding protein experiments with binding sites
2. ATtRACT - RNA-binding protein motifs database (real experimentally validated motifs)
3. RNAcentral - Real non-coding RNA sequences

This script provides REAL RNA data for protein-RNA interaction models.
NO synthetic/generated sequences are created.

Usage:
    python -m download.download_rna --source encode
    python -m download.download_rna --source attract
    python -m download.download_rna --source rnacentral
    python -m download.download_rna --fasta-file rna_sequences.fasta

Data Sources:
    - ENCODE eCLIP: Real experimentally determined RBP binding sites
    - ATtRACT: Experimentally validated RNA-binding protein motifs
    - RNAcentral: Real non-coding RNA sequences
"""

import argparse
import csv
import gzip
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

# ENCODE API
ENCODE_API_URL = "https://www.encodeproject.org"

# ATtRACT database
ATTRACT_URL = "https://attract.cnic.es/attract/static/ATtRACT.zip"

# RNAcentral API
RNACENTRAL_API_URL = "https://rnacentral.org/api/v1"
RNACENTRAL_SEARCH_URL = "https://rnacentral.org/api/v1/rna"

# Ensembl FTP for bulk ncRNA downloads (more reliable than RNAcentral API)
ENSEMBL_FTP_URL = "https://ftp.ensembl.org/pub/current_fasta"


def create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 1.0,
    timeout: int = 60,
) -> requests.Session:
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =============================================================================
# RNA VOCABULARY AND UTILITIES
# =============================================================================

RNA_NUCLEOTIDES = set("ACGU")


def is_valid_rna(sequence: str) -> bool:
    """Check if sequence is valid RNA"""
    if not sequence:
        return False
    seq_upper = sequence.upper().replace("T", "U")
    return all(c in RNA_NUCLEOTIDES or c == "N" for c in seq_upper)


def dna_to_rna(sequence: str) -> str:
    """Convert DNA sequence to RNA (T -> U)"""
    return sequence.upper().replace("T", "U")


# =============================================================================
# ENCODE eCLIP DATA DOWNLOAD
# =============================================================================


def download_encode_eclip(
    output_dir: Path,
    organism: str = "Homo sapiens",
    max_experiments: int = 100,
    max_binding_sites: int = 100000,
) -> Dict[str, List[Dict]]:
    """
    Download real eCLIP data from ENCODE

    eCLIP (enhanced CrossLinking and ImmunoPrecipitation) provides
    experimentally determined RNA binding sites for RNA-binding proteins.

    Args:
        output_dir: Output directory
        organism: Target organism
        max_experiments: Maximum number of experiments to download
        max_binding_sites: Maximum binding sites per RBP

    Returns:
        Dictionary mapping RBP names to binding site data
    """
    print("=" * 70)
    print("DOWNLOADING ENCODE eCLIP DATA")
    print("=" * 70)
    print(f"Organism: {organism}")
    print(f"Max experiments: {max_experiments}")
    print()

    encode_dir = output_dir / "encode"
    encode_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Accept": "application/json"}

    # Search for eCLIP experiments
    print("Searching for eCLIP experiments...")

    search_url = (
        f"{ENCODE_API_URL}/search/"
        f"?type=Experiment"
        f"&assay_title=eCLIP"
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
    print(f"✓ Found {len(experiments)} eCLIP experiments")

    if not experiments:
        print("No experiments found. Try different search parameters.")
        return {}

    # Process each experiment
    rbp_binding_sites: Dict[str, List[Dict]] = {}
    all_binding_sites: List[Dict] = []

    for exp in tqdm(experiments, desc="Processing experiments"):
        exp_accession = exp.get("accession", "")

        # Get target (RBP) information
        targets = exp.get("target", [])
        if isinstance(targets, str):
            targets = [targets]

        # Get the RBP name from the target
        rbp_name = None
        for target in targets:
            if isinstance(target, dict):
                rbp_name = target.get("label", "").upper()
            elif isinstance(target, str):
                # Target is a path like "/targets/RBFOX2-human/"
                parts = target.strip("/").split("/")
                if len(parts) >= 2:
                    rbp_name = parts[-1].split("-")[0].upper()
            if rbp_name:
                break

        if not rbp_name:
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

                    sites_for_rbp = []
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
                            "rbp": rbp_name,
                            "chromosome": chrom,
                            "start": start,
                            "end": end,
                            "strand": strand,
                            "score": score,
                            "name": name,
                            "experiment": exp_accession,
                            "source": "ENCODE_eCLIP",
                        }

                        sites_for_rbp.append(site_data)
                        all_binding_sites.append(site_data)

                    if sites_for_rbp:
                        if rbp_name not in rbp_binding_sites:
                            rbp_binding_sites[rbp_name] = []
                        rbp_binding_sites[rbp_name].extend(sites_for_rbp)

                    # Only process one BED file per experiment
                    break

                except requests.RequestException:
                    continue

        # Rate limiting
        time.sleep(0.1)

    print(f"\n✓ Downloaded binding sites for {len(rbp_binding_sites)} RBPs")
    print(f"✓ Total binding sites: {len(all_binding_sites):,}")

    # Save results
    if rbp_binding_sites:
        # Save all binding sites
        sites_file = encode_dir / "binding_sites.json"
        with open(sites_file, "w") as f:
            json.dump(rbp_binding_sites, f, indent=2)
        print(f"✓ Saved: {sites_file}")

        # Save as TSV
        tsv_file = encode_dir / "binding_sites.tsv"
        with open(tsv_file, "w") as f:
            f.write("rbp\tchromosome\tstart\tend\tstrand\tscore\texperiment\n")
            for site in all_binding_sites:
                f.write(
                    f"{site['rbp']}\t{site['chromosome']}\t{site['start']}\t"
                    f"{site['end']}\t{site['strand']}\t{site['score']}\t"
                    f"{site['experiment']}\n"
                )
        print(f"✓ Saved: {tsv_file}")

        # Save summary
        summary_file = encode_dir / "summary.json"
        summary = {
            "total_rbps": len(rbp_binding_sites),
            "total_binding_sites": len(all_binding_sites),
            "rbp_counts": {rbp: len(sites) for rbp, sites in rbp_binding_sites.items()},
            "organism": organism,
            "source": "ENCODE_eCLIP",
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {summary_file}")

    return rbp_binding_sites


# =============================================================================
# ATtRACT DATABASE DOWNLOAD
# =============================================================================


def download_attract_database(output_dir: Path) -> Dict[str, List[Dict]]:
    """
    Download and parse the ATtRACT database

    ATtRACT contains experimentally validated RNA-binding protein motifs
    from published literature.

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
    db_txt_file = "ATtRACT_db.txt"

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
    else:
        print(f"✓ Using cached: {db_file}")

    # Parse the database
    print("\nParsing ATtRACT database...")
    rbp_motifs: Dict[str, List[Dict]] = {}
    all_motifs: List[Dict] = []

    try:
        with zipfile.ZipFile(db_file, "r") as zip_file:
            path = zipfile.Path(zip_file, at=db_txt_file)
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")

                # Skip header
                next(reader, None)

                for row in reader:
                    if len(row) < 5:
                        continue

                    gene_name = row[0].upper()
                    gene_id = row[1] if len(row) > 1 else ""
                    organism = row[3] if len(row) > 3 else ""
                    motif = row[4] if len(row) > 4 else ""
                    pubmed_id = row[8] if len(row) > 8 else ""

                    # Filter for valid motifs
                    if not motif or len(motif) < 3:
                        continue

                    motif_data = {
                        "gene_name": gene_name,
                        "gene_id": gene_id,
                        "organism": organism,
                        "motif": motif,
                        "pubmed_id": pubmed_id,
                        "source": "ATtRACT",
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
# ENSEMBL ncRNA SEQUENCES (RELIABLE BULK DOWNLOAD)
# =============================================================================


def download_ensembl_ncrna(
    output_dir: Path,
    organism: str = "homo_sapiens",
    max_sequences: int = 10000,
) -> List[Dict]:
    """
    Download real ncRNA sequences from Ensembl FTP

    This is more reliable than the RNAcentral API for bulk downloads.
    Ensembl provides high-quality curated ncRNA sequences.

    Args:
        output_dir: Output directory
        organism: Target organism (lowercase with underscore, e.g., 'homo_sapiens')
        max_sequences: Maximum sequences to download

    Returns:
        List of sequence dictionaries
    """
    print("=" * 70)
    print("DOWNLOADING Ensembl ncRNA SEQUENCES")
    print("=" * 70)
    print(f"Organism: {organism}")
    print(f"Max sequences: {max_sequences}")
    print()

    ensembl_dir = output_dir / "ensembl"
    ensembl_dir.mkdir(parents=True, exist_ok=True)

    # Convert organism name to Ensembl format
    organism_lower = organism.lower().replace(" ", "_")

    # Ensembl ncRNA FASTA URL
    ncrna_url = f"{ENSEMBL_FTP_URL}/{organism_lower}/ncrna/{organism_lower.capitalize()}.*.ncrna.fa.gz"

    # Try to find the correct file
    print(f"Searching for ncRNA file for {organism_lower}...")

    # Common Ensembl assembly names
    assemblies = {
        "homo_sapiens": "GRCh38",
        "mus_musculus": "GRCm39",
        "danio_rerio": "GRCz11",
        "drosophila_melanogaster": "BDGP6",
        "caenorhabditis_elegans": "WBcel235",
        "saccharomyces_cerevisiae": "R64-1-1",
    }

    assembly = assemblies.get(organism_lower, "")

    # Construct URL
    if assembly:
        ncrna_url = f"{ENSEMBL_FTP_URL}/{organism_lower}/ncrna/{organism_lower.capitalize()}.{assembly}.ncrna.fa.gz"
    else:
        # Try to list directory and find file
        print(f"  Unknown assembly for {organism_lower}, trying default...")
        ncrna_url = f"{ENSEMBL_FTP_URL}/{organism_lower}/ncrna/"

    all_sequences: List[Dict] = []

    try:
        # Download the file
        print(f"Downloading from: {ncrna_url}")

        session = create_session_with_retries(
            retries=3, backoff_factor=2.0, timeout=300
        )
        response = session.get(ncrna_url, stream=True, timeout=300)
        response.raise_for_status()

        # Save and decompress
        gz_file = ensembl_dir / f"{organism_lower}_ncrna.fa.gz"
        total_size = int(response.headers.get("content-length", 0))

        with open(gz_file, "wb") as f:
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

        print(f"✓ Downloaded: {gz_file}")

        # Parse the gzipped FASTA file
        print("Parsing ncRNA sequences...")

        import gzip

        current_id = None
        current_seq = []
        current_meta = {}

        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence
                    if current_id and current_seq:
                        seq = "".join(current_seq)
                        seq = dna_to_rna(seq)  # Convert T to U

                        if is_valid_rna(seq) and len(seq) >= 20:
                            all_sequences.append(
                                {
                                    "id": current_id,
                                    "sequence": seq,
                                    "length": len(seq),
                                    "rna_type": current_meta.get("biotype", "ncRNA"),
                                    "description": current_meta.get("description", ""),
                                    "gene_name": current_meta.get("gene", ""),
                                    "organism": organism,
                                    "source": "Ensembl",
                                }
                            )

                            if len(all_sequences) >= max_sequences:
                                break

                    # Parse header: >ENST00000456328.2 ncrna chromosome:GRCh38:1:... gene:ENSG00000223972.5 gene_biotype:...
                    header = line[1:]
                    parts = header.split()
                    current_id = parts[0] if parts else ""

                    current_meta = {}
                    for part in parts[1:]:
                        if ":" in part:
                            key, value = part.split(":", 1)
                            current_meta[key] = value
                        elif part.startswith("gene_biotype"):
                            current_meta["biotype"] = part.split(":")[-1]

                    # Get biotype from gene_biotype field
                    if "gene_biotype" in current_meta:
                        current_meta["biotype"] = current_meta["gene_biotype"]

                    current_seq = []
                else:
                    current_seq.append(line)

                if len(all_sequences) >= max_sequences:
                    break

        # Don't forget last sequence
        if current_id and current_seq and len(all_sequences) < max_sequences:
            seq = "".join(current_seq)
            seq = dna_to_rna(seq)
            if is_valid_rna(seq) and len(seq) >= 20:
                all_sequences.append(
                    {
                        "id": current_id,
                        "sequence": seq,
                        "length": len(seq),
                        "rna_type": current_meta.get("biotype", "ncRNA"),
                        "description": current_meta.get("description", ""),
                        "gene_name": current_meta.get("gene", ""),
                        "organism": organism,
                        "source": "Ensembl",
                    }
                )

        print(f"✓ Parsed {len(all_sequences):,} ncRNA sequences")

    except requests.RequestException as e:
        print(f"❌ Failed to download Ensembl ncRNA: {e}")
        print("  Try using --source attract or --source encode instead")
        return []

    # Save sequences
    if all_sequences:
        # Save as JSON
        json_file = ensembl_dir / "ncrna_sequences.json"
        with open(json_file, "w") as f:
            json.dump(all_sequences, f, indent=2)
        print(f"✓ Saved: {json_file}")

        # Save as FASTA
        fasta_file = ensembl_dir / "ncrna_sequences.fasta"
        with open(fasta_file, "w") as f:
            for seq_data in all_sequences:
                f.write(f">{seq_data['id']}|{seq_data['rna_type']}\n")
                seq = seq_data["sequence"]
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")
        print(f"✓ Saved: {fasta_file}")

        # Save as TSV
        tsv_file = ensembl_dir / "ncrna_sequences.tsv"
        with open(tsv_file, "w") as f:
            f.write("id\trna_type\tgene_name\tlength\tsequence\n")
            for seq_data in all_sequences:
                f.write(
                    f"{seq_data['id']}\t{seq_data['rna_type']}\t"
                    f"{seq_data.get('gene_name', '')}\t{seq_data['length']}\t"
                    f"{seq_data['sequence']}\n"
                )
        print(f"✓ Saved: {tsv_file}")

        # Print summary by RNA type
        rna_types = {}
        for seq in all_sequences:
            rna_type = seq.get("rna_type", "unknown")
            rna_types[rna_type] = rna_types.get(rna_type, 0) + 1

        print("\nSequences by RNA type:")
        for rna_type, count in sorted(rna_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  • {rna_type}: {count:,}")

    return all_sequences


# =============================================================================
# RNACENTRAL - REAL RNA SEQUENCES (API - may be slow)
# =============================================================================


def download_rnacentral_sequences(
    output_dir: Path,
    rna_types: Optional[List[str]] = None,
    organism: str = "Homo sapiens",
    max_sequences: int = 10000,
) -> List[Dict]:
    """
    Download real RNA sequences from RNAcentral

    RNAcentral is a comprehensive database of non-coding RNA sequences
    from multiple source databases.

    Args:
        output_dir: Output directory
        rna_types: Types of RNA to download (e.g., ["mRNA", "lncRNA", "miRNA"])
        organism: Target organism
        max_sequences: Maximum sequences to download

    Returns:
        List of sequence dictionaries
    """
    print("=" * 70)
    print("DOWNLOADING RNAcentral SEQUENCES")
    print("=" * 70)
    print(f"Organism: {organism}")
    print(f"Max sequences: {max_sequences}")
    print()

    if rna_types is None:
        rna_types = ["lncRNA", "mRNA", "miRNA", "snoRNA", "snRNA"]

    rnacentral_dir = output_dir / "rnacentral"
    rnacentral_dir.mkdir(parents=True, exist_ok=True)

    all_sequences: List[Dict] = []

    # Create session with retry logic
    session = create_session_with_retries(retries=3, backoff_factor=2.0, timeout=120)
    headers = {"Accept": "application/json"}

    for rna_type in rna_types:
        print(f"\nSearching for {rna_type}...")

        # Build search parameters
        params = {
            "rna_type": rna_type,
            "species": organism,
            "page_size": 100,
        }

        sequences_for_type = []
        page = 1
        max_retries_per_page = 3
        target_count = max_sequences // len(rna_types)

        while len(sequences_for_type) < target_count:
            retry_count = 0
            success = False

            while retry_count < max_retries_per_page and not success:
                try:
                    params["page"] = page
                    response = session.get(
                        RNACENTRAL_SEARCH_URL,
                        params=params,
                        headers=headers,
                        timeout=120,
                    )

                    if response.status_code == 404:
                        # No more pages
                        success = True
                        break

                    if response.status_code != 200:
                        print(f"  Warning: HTTP {response.status_code}, retrying...")
                        retry_count += 1
                        time.sleep(2**retry_count)
                        continue

                    data = response.json()
                    results = data.get("results", [])

                    if not results:
                        success = True
                        break

                    for entry in results:
                        rna_id = entry.get("rnacentral_id", "")
                        sequence = entry.get("sequence", "")
                        description = entry.get("description", "")

                        if sequence and len(sequence) >= 20:
                            # Convert to RNA if needed
                            sequence = dna_to_rna(sequence)

                            if is_valid_rna(sequence):
                                sequences_for_type.append(
                                    {
                                        "id": rna_id,
                                        "sequence": sequence,
                                        "length": len(sequence),
                                        "rna_type": rna_type,
                                        "description": description,
                                        "organism": organism,
                                        "source": "RNAcentral",
                                    }
                                )

                                if len(sequences_for_type) >= target_count:
                                    break

                    success = True
                    page += 1
                    time.sleep(0.5)  # Rate limiting

                    if not data.get("next"):
                        break

                except requests.exceptions.Timeout:
                    retry_count += 1
                    wait_time = 2**retry_count
                    print(
                        f"  Timeout, waiting {wait_time}s before retry {retry_count}/{max_retries_per_page}..."
                    )
                    time.sleep(wait_time)

                except requests.RequestException as e:
                    retry_count += 1
                    print(
                        f"  Warning: {e}, retry {retry_count}/{max_retries_per_page}..."
                    )
                    time.sleep(2**retry_count)

            if not success:
                print(
                    f"  Failed to fetch page {page} after {max_retries_per_page} retries"
                )
                break

        print(f"  ✓ Downloaded {len(sequences_for_type)} {rna_type} sequences")
        all_sequences.extend(sequences_for_type)

    print(f"\n✓ Total sequences downloaded: {len(all_sequences):,}")

    # Save sequences
    if all_sequences:
        # Save as JSON
        json_file = rnacentral_dir / "rna_sequences.json"
        with open(json_file, "w") as f:
            json.dump(all_sequences, f, indent=2)
        print(f"✓ Saved: {json_file}")

        # Save as FASTA
        fasta_file = rnacentral_dir / "rna_sequences.fasta"
        with open(fasta_file, "w") as f:
            for seq_data in all_sequences:
                f.write(f">{seq_data['id']}|{seq_data['rna_type']}\n")
                # Write sequence in lines of 80 characters
                seq = seq_data["sequence"]
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")
        print(f"✓ Saved: {fasta_file}")

        # Save as TSV
        tsv_file = rnacentral_dir / "rna_sequences.tsv"
        with open(tsv_file, "w") as f:
            f.write("id\trna_type\tlength\tsequence\n")
            for seq_data in all_sequences:
                f.write(
                    f"{seq_data['id']}\t{seq_data['rna_type']}\t"
                    f"{seq_data['length']}\t{seq_data['sequence']}\n"
                )
        print(f"✓ Saved: {tsv_file}")

    return all_sequences


# =============================================================================
# FILE I/O
# =============================================================================


def load_fasta_file(fasta_path: Path) -> List[Dict]:
    """
    Load RNA sequences from a FASTA file

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
                    seq = "".join(current_seq)
                    seq = dna_to_rna(seq)  # Convert T to U
                    if is_valid_rna(seq):
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
        seq = "".join(current_seq)
        seq = dna_to_rna(seq)
        if is_valid_rna(seq):
            sequences.append(
                {
                    "id": current_id,
                    "sequence": seq,
                    "length": len(seq),
                    "source": "fasta",
                    **current_meta,
                }
            )

    print(f"✓ Loaded {len(sequences):,} valid RNA sequences")
    return sequences


def save_sequences(
    sequences: List[Dict],
    output_dir: Path,
    rbp_motifs: Optional[Dict] = None,
) -> None:
    """
    Save RNA sequences to multiple formats

    Args:
        sequences: List of sequence dictionaries
        output_dir: Output directory
        rbp_motifs: Optional RBP motif dictionary
    """
    if not sequences:
        print("No sequences to save.")
        return

    print(f"\nSaving {len(sequences):,} sequences...")

    # Save as JSON
    json_file = output_dir / "rna_sequences.json"
    with open(json_file, "w") as f:
        json.dump(sequences, f, indent=2)
    print(f"✓ Saved: {json_file}")

    # Save as FASTA
    fasta_file = output_dir / "rna_sequences.fasta"
    with open(fasta_file, "w") as f:
        for seq_data in sequences:
            seq_id = seq_data.get("id", "unknown")
            rbp = seq_data.get("rbp", "")
            header = f">{seq_id}"
            if rbp:
                header += f"|rbp={rbp}"
            f.write(header + "\n")

            # Write sequence in lines of 80 characters
            seq = seq_data["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
    print(f"✓ Saved: {fasta_file}")

    # Save as TSV
    tsv_file = output_dir / "rna_sequences.tsv"
    with open(tsv_file, "w") as f:
        f.write("id\trbp\tlength\tsource\tsequence\n")
        for seq_data in sequences:
            f.write(
                f"{seq_data.get('id', 'unknown')}\t"
                f"{seq_data.get('rbp', '')}\t"
                f"{seq_data.get('length', len(seq_data['sequence']))}\t"
                f"{seq_data.get('source', 'unknown')}\t"
                f"{seq_data['sequence']}\n"
            )
    print(f"✓ Saved: {tsv_file}")

    # Save sequences by RBP
    rbp_sequences: Dict[str, List[Dict]] = {}
    for seq_data in sequences:
        rbp = seq_data.get("rbp")
        if rbp:
            if rbp not in rbp_sequences:
                rbp_sequences[rbp] = []
            rbp_sequences[rbp].append(seq_data)

    if rbp_sequences:
        by_rbp_dir = output_dir / "by_rbp"
        by_rbp_dir.mkdir(exist_ok=True)

        for rbp, seqs in rbp_sequences.items():
            rbp_file = by_rbp_dir / f"{rbp}_sequences.fasta"
            with open(rbp_file, "w") as f:
                for i, seq_data in enumerate(seqs):
                    f.write(f">{rbp}_{i}\n")
                    seq = seq_data["sequence"]
                    for j in range(0, len(seq), 80):
                        f.write(seq[j : j + 80] + "\n")

        print(f"✓ Saved {len(rbp_sequences)} RBP-specific files to {by_rbp_dir}")


def save_metadata(
    output_dir: Path,
    source: str,
    sequence_count: int,
    rbp_count: int = 0,
    motif_count: int = 0,
    binding_site_count: int = 0,
) -> None:
    """Save download metadata"""
    output_file = output_dir / "download_info.json"

    info = {
        "source": source,
        "sequence_count": sequence_count,
        "rbp_count": rbp_count,
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
    rbp_motifs: Optional[Dict] = None,
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

    if rbp_motifs:
        print(f"RBPs with motifs: {len(rbp_motifs):,}")
        total_motifs = sum(len(v) for v in rbp_motifs.values())
        print(f"Total motifs: {total_motifs:,}")

    if binding_sites:
        print(f"RBPs with binding sites: {len(binding_sites):,}")
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
        description="Download RNA Data (Real Data Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m download.download_rna --source encode
    python -m download.download_rna --source attract
    python -m download.download_rna --source ensembl
    python -m download.download_rna --source rnacentral
    python -m download.download_rna --source all
    python -m download.download_rna --fasta-file rna_sequences.fasta

Sources:
    encode     - ENCODE eCLIP data (real RBP binding sites)
    attract    - ATtRACT database (experimentally validated RBP motifs)
    ensembl    - Ensembl ncRNA sequences (recommended, fast bulk download)
    rnacentral - RNAcentral API (real ncRNA sequences, may be slow)
    all        - Download from all sources
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="attract",
        choices=["encode", "attract", "ensembl", "rnacentral", "all"],
        help="Data source (default: attract)",
    )

    parser.add_argument(
        "--fasta-file",
        type=str,
        help="Load RNA sequences from FASTA file",
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
        "--max-sequences",
        type=int,
        default=10000,
        help="Maximum sequences to download from RNAcentral (default: 10000)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rna",
        help="Output directory (default: data/rna)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RNA DATA DOWNLOAD (REAL DATA ONLY)")
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
    rbp_motifs: Optional[Dict] = None
    binding_sites: Optional[Dict] = None

    # Load from FASTA file if specified
    if args.fasta_file:
        sequences = load_fasta_file(Path(args.fasta_file))

    elif args.source == "encode" or args.source == "all":
        # Download ENCODE eCLIP data
        binding_sites = download_encode_eclip(
            output_dir,
            organism=args.organism,
            max_experiments=args.max_experiments,
        )

        if args.source == "all":
            # Also download ATtRACT
            rbp_motifs = download_attract_database(output_dir)

            # Also download Ensembl ncRNA (more reliable than RNAcentral API)
            sequences = download_ensembl_ncrna(
                output_dir,
                organism=args.organism,
                max_sequences=args.max_sequences,
            )

    elif args.source == "attract":
        rbp_motifs = download_attract_database(output_dir)

    elif args.source == "ensembl":
        # Download from Ensembl (recommended for bulk ncRNA)
        sequences = download_ensembl_ncrna(
            output_dir,
            organism=args.organism,
            max_sequences=args.max_sequences,
        )

    elif args.source == "rnacentral":
        # Download from RNAcentral API (may be slow/timeout)
        print("Note: RNAcentral API can be slow. Consider --source ensembl instead.")
        sequences = download_rnacentral_sequences(
            output_dir,
            organism=args.organism,
            max_sequences=args.max_sequences,
        )

    # Save sequences if we have them
    if sequences:
        save_sequences(sequences, output_dir, rbp_motifs)

    # Save metadata
    save_metadata(
        output_dir=output_dir,
        source=args.source if not args.fasta_file else "fasta",
        sequence_count=len(sequences),
        rbp_count=len(rbp_motifs) if rbp_motifs else 0,
        motif_count=sum(len(v) for v in rbp_motifs.values()) if rbp_motifs else 0,
        binding_site_count=sum(len(v) for v in binding_sites.values())
        if binding_sites
        else 0,
    )

    # Print summary
    print_summary(output_dir, sequences, rbp_motifs, binding_sites)

    print("Done!")
    print()
    print("This RNA data can be used with:")
    print("  • python -m prepare.prepare_p2r_data - Prepare protein-RNA data")
    print()


if __name__ == "__main__":
    main()
