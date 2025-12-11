#!/usr/bin/env python3
"""
Build Protein-Molecule Interactions
====================================

This script creates protein-molecule interaction data by matching:
1. ChEMBL activities (IC50, Ki, etc.) with target proteins
2. Target proteins with their UniProt sequences
3. Molecules with their SMILES representations

This creates REAL interaction pairs based on experimentally validated
binding data from the ChEMBL database.

Usage:
    python -m download.build_p2m_interactions
    python -m download.build_p2m_interactions --protein-dir data/proteins/homo_sapiens
    python -m download.build_p2m_interactions --molecule-dir data/molecules

Output:
    protein_molecule_interactions.tsv - TSV file with protein-molecule interaction pairs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def load_proteins(protein_dir: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Load protein data from UniProt download

    Args:
        protein_dir: Directory containing protein data

    Returns:
        Tuple of (proteins_by_gene, proteins_by_uniprot)
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

            if sequence:
                protein_info = {
                    "uniprot_id": uniprot_id,
                    "gene_name": gene_name,
                    "sequence": sequence,
                    "length": len(sequence),
                }
                proteins_by_uniprot[uniprot_id] = protein_info
                if gene_name:
                    proteins_by_gene[gene_name] = protein_info

        print(f"  ✓ Loaded {len(proteins_by_uniprot):,} proteins")

    return proteins_by_gene, proteins_by_uniprot


def load_molecules(molecule_dir: Path) -> Dict[str, Dict]:
    """
    Load molecule data from ChEMBL download

    Args:
        molecule_dir: Directory containing molecule data

    Returns:
        Dictionary mapping molecule IDs to molecule info
    """
    print("Loading molecules...")

    molecules: Dict[str, Dict] = {}

    # Load from molecules.json
    json_file = molecule_dir / "molecules.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                mol_id = entry.get("molecule_id", entry.get("chembl_id", ""))
                smiles = entry.get("smiles", entry.get("canonical_smiles", ""))
                if mol_id and smiles:
                    molecules[mol_id] = {
                        "molecule_id": mol_id,
                        "smiles": smiles,
                        "molecular_weight": entry.get("molecular_weight"),
                    }
        elif isinstance(data, dict):
            for mol_id, info in data.items():
                smiles = info.get("smiles", info.get("canonical_smiles", ""))
                if smiles:
                    molecules[mol_id] = {
                        "molecule_id": mol_id,
                        "smiles": smiles,
                        "molecular_weight": info.get("molecular_weight"),
                    }

        print(f"  ✓ Loaded {len(molecules):,} molecules from JSON")

    # Also try loading from TSV
    tsv_file = molecule_dir / "molecules.tsv"
    if tsv_file.exists() and not molecules:
        with open(tsv_file) as f:
            header = f.readline().strip().lower().split("\t")

            mol_id_idx = None
            smiles_idx = None

            for i, col in enumerate(header):
                if "molecule" in col and "id" in col:
                    mol_id_idx = i
                elif col == "smiles" or "canonical" in col:
                    smiles_idx = i

            if mol_id_idx is not None and smiles_idx is not None:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) > max(mol_id_idx, smiles_idx):
                        mol_id = parts[mol_id_idx]
                        smiles = parts[smiles_idx]
                        if mol_id and smiles:
                            molecules[mol_id] = {
                                "molecule_id": mol_id,
                                "smiles": smiles,
                            }

        print(f"  ✓ Loaded {len(molecules):,} molecules from TSV")

    # Load from SMILES file
    smi_file = molecule_dir / "molecules.smi"
    if smi_file.exists() and not molecules:
        with open(smi_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles = parts[0]
                    mol_id = parts[1]
                    molecules[mol_id] = {
                        "molecule_id": mol_id,
                        "smiles": smiles,
                    }

        print(f"  ✓ Loaded {len(molecules):,} molecules from SMILES file")

    return molecules


def load_targets(molecule_dir: Path) -> Dict[str, Dict]:
    """
    Load target data from ChEMBL download

    Args:
        molecule_dir: Directory containing molecule/target data

    Returns:
        Dictionary mapping target IDs to target info
    """
    print("Loading targets...")

    targets: Dict[str, Dict] = {}

    tsv_file = molecule_dir / "targets.tsv"
    if tsv_file.exists():
        with open(tsv_file) as f:
            header = f.readline().strip().lower().split("\t")

            # Find column indices
            target_id_idx = None
            uniprot_idx = None
            name_idx = None
            organism_idx = None

            for i, col in enumerate(header):
                if "target" in col and "id" in col:
                    target_id_idx = i
                elif "uniprot" in col:
                    uniprot_idx = i
                elif "name" in col or "pref" in col:
                    name_idx = i
                elif "organism" in col:
                    organism_idx = i

            if target_id_idx is not None:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) > target_id_idx:
                        target_id = parts[target_id_idx]
                        targets[target_id] = {
                            "target_id": target_id,
                            "uniprot_id": parts[uniprot_idx]
                            if uniprot_idx and len(parts) > uniprot_idx
                            else "",
                            "name": parts[name_idx]
                            if name_idx and len(parts) > name_idx
                            else "",
                            "organism": parts[organism_idx]
                            if organism_idx and len(parts) > organism_idx
                            else "",
                        }

        print(f"  ✓ Loaded {len(targets):,} targets")

    return targets


def load_activities(molecule_dir: Path) -> List[Dict]:
    """
    Load activity data from ChEMBL download

    Args:
        molecule_dir: Directory containing activity data

    Returns:
        List of activity dictionaries
    """
    print("Loading activities...")

    activities: List[Dict] = []

    tsv_file = molecule_dir / "activities.tsv"
    if tsv_file.exists():
        with open(tsv_file) as f:
            header = f.readline().strip().lower().split("\t")

            # Find column indices
            mol_idx = None
            target_idx = None
            type_idx = None
            value_idx = None
            units_idx = None
            pchembl_idx = None

            for i, col in enumerate(header):
                if "molecule" in col and "id" in col:
                    mol_idx = i
                elif "target" in col and "id" in col:
                    target_idx = i
                elif "type" in col:
                    type_idx = i
                elif col == "standard_value" or col == "value":
                    value_idx = i
                elif "unit" in col:
                    units_idx = i
                elif "pchembl" in col:
                    pchembl_idx = i

            for line in f:
                parts = line.strip().split("\t")

                try:
                    activity = {
                        "molecule_id": parts[mol_idx]
                        if mol_idx is not None and len(parts) > mol_idx
                        else "",
                        "target_id": parts[target_idx]
                        if target_idx is not None and len(parts) > target_idx
                        else "",
                        "activity_type": parts[type_idx]
                        if type_idx is not None and len(parts) > type_idx
                        else "",
                        "value": float(parts[value_idx])
                        if value_idx is not None
                        and len(parts) > value_idx
                        and parts[value_idx]
                        else None,
                        "units": parts[units_idx]
                        if units_idx is not None and len(parts) > units_idx
                        else "",
                        "pchembl_value": float(parts[pchembl_idx])
                        if pchembl_idx is not None
                        and len(parts) > pchembl_idx
                        and parts[pchembl_idx]
                        else None,
                    }

                    if activity["molecule_id"] and activity["target_id"]:
                        activities.append(activity)

                except (ValueError, IndexError):
                    continue

        print(f"  ✓ Loaded {len(activities):,} activities")

    return activities


def build_interactions(
    proteins_by_uniprot: Dict[str, Dict],
    molecules: Dict[str, Dict],
    targets: Dict[str, Dict],
    activities: List[Dict],
    min_pchembl: float = 5.0,
    activity_types: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Build protein-molecule interaction pairs

    Args:
        proteins_by_uniprot: Proteins indexed by UniProt ID
        molecules: Molecules indexed by molecule ID
        targets: Targets indexed by target ID
        activities: List of activity data
        min_pchembl: Minimum pChEMBL value for active compounds
        activity_types: Set of activity types to include (e.g., {'IC50', 'Ki'})

    Returns:
        List of interaction dictionaries
    """
    print("\nBuilding protein-molecule interactions...")

    if activity_types is None:
        activity_types = {"IC50", "Ki", "Kd", "EC50"}

    interactions = []
    seen_pairs = set()

    matched_targets = 0
    unmatched_targets = set()

    for activity in activities:
        molecule_id = activity["molecule_id"]
        target_id = activity["target_id"]
        activity_type = activity["activity_type"]
        pchembl = activity["pchembl_value"]

        # Filter by activity type
        if activity_type not in activity_types:
            continue

        # Filter by pChEMBL value (binding affinity)
        if pchembl is None or pchembl < min_pchembl:
            continue

        # Get molecule SMILES
        if molecule_id not in molecules:
            continue
        smiles = molecules[molecule_id]["smiles"]

        # Get target UniProt ID
        if target_id not in targets:
            continue
        uniprot_id = targets[target_id].get("uniprot_id", "")

        if not uniprot_id:
            unmatched_targets.add(target_id)
            continue

        # Get protein sequence
        if uniprot_id not in proteins_by_uniprot:
            unmatched_targets.add(target_id)
            continue

        matched_targets += 1
        protein_info = proteins_by_uniprot[uniprot_id]
        protein_seq = protein_info["sequence"]

        # Avoid duplicate pairs
        pair_key = (uniprot_id, molecule_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Create interaction
        interaction = {
            "protein_id": uniprot_id,
            "protein_seq": protein_seq,
            "smiles": smiles,
            "label": 1,
            "affinity": pchembl,
            "activity_type": activity_type,
            "source": "ChEMBL",
            "molecule_id": molecule_id,
            "target_name": targets[target_id].get("name", ""),
        }
        interactions.append(interaction)

    print(f"  ✓ Matched {matched_targets:,} activities to proteins")
    print(f"  ✓ Could not match {len(unmatched_targets):,} targets")
    print(f"  ✓ Created {len(interactions):,} protein-molecule interactions")

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
    output_file = output_dir / "protein_molecule_interactions.tsv"

    with open(output_file, "w") as f:
        # Header
        f.write(
            "protein_id\tprotein_seq\tsmiles\tlabel\taffinity\tactivity_type\tsource\tmolecule_id\ttarget_name\n"
        )

        for interaction in interactions:
            f.write(
                f"{interaction['protein_id']}\t"
                f"{interaction['protein_seq']}\t"
                f"{interaction['smiles']}\t"
                f"{interaction['label']}\t"
                f"{interaction['affinity']}\t"
                f"{interaction['activity_type']}\t"
                f"{interaction['source']}\t"
                f"{interaction['molecule_id']}\t"
                f"{interaction['target_name']}\n"
            )

    print(f"✓ Saved: {output_file}")

    # Also save as JSON for easier loading
    json_file = output_dir / "protein_molecule_interactions.json"
    with open(json_file, "w") as f:
        json.dump(interactions, f, indent=2)
    print(f"✓ Saved: {json_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("INTERACTION SUMMARY")
    print("=" * 60)

    unique_proteins = len(set(i["protein_id"] for i in interactions))
    unique_molecules = len(set(i["molecule_id"] for i in interactions))

    print(f"Total interactions: {len(interactions):,}")
    print(f"Unique proteins: {unique_proteins:,}")
    print(f"Unique molecules: {unique_molecules:,}")

    # Affinity distribution
    affinities = [i["affinity"] for i in interactions if i["affinity"]]
    if affinities:
        print(f"pChEMBL range: {min(affinities):.2f} - {max(affinities):.2f}")
        print(f"Mean pChEMBL: {sum(affinities) / len(affinities):.2f}")

    # Activity type distribution
    activity_types = {}
    for i in interactions:
        at = i["activity_type"]
        activity_types[at] = activity_types.get(at, 0) + 1

    print("\nBy activity type:")
    for at, count in sorted(activity_types.items(), key=lambda x: -x[1]):
        print(f"  • {at}: {count:,}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Build Protein-Molecule Interactions from ChEMBL Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m download.build_p2m_interactions
    python -m download.build_p2m_interactions --protein-dir data/proteins/homo_sapiens
    python -m download.build_p2m_interactions --min-pchembl 6.0
        """,
    )

    parser.add_argument(
        "--protein-dir",
        type=str,
        default="data/proteins/homo_sapiens",
        help="Directory containing protein data (default: data/proteins/homo_sapiens)",
    )

    parser.add_argument(
        "--molecule-dir",
        type=str,
        default="data/molecules",
        help="Directory containing molecule data (default: data/molecules)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/molecules",
        help="Output directory (default: data/molecules)",
    )

    parser.add_argument(
        "--min-pchembl",
        type=float,
        default=5.0,
        help="Minimum pChEMBL value for active compounds (default: 5.0)",
    )

    parser.add_argument(
        "--activity-types",
        type=str,
        nargs="+",
        default=["IC50", "Ki", "Kd", "EC50"],
        help="Activity types to include (default: IC50 Ki Kd EC50)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BUILD PROTEIN-MOLECULE INTERACTIONS")
    print("=" * 70)
    print()
    print("This script builds protein-molecule interaction pairs from:")
    print("  • UniProt protein sequences")
    print("  • ChEMBL bioactivity data (IC50, Ki, etc.)")
    print()

    protein_dir = Path(args.protein_dir)
    molecule_dir = Path(args.molecule_dir)
    output_dir = Path(args.output_dir)

    # Check directories exist
    if not protein_dir.exists():
        print(f"❌ Protein directory not found: {protein_dir}")
        print("   Run: python -m download.download_pro --species 'Homo sapiens'")
        sys.exit(1)

    if not molecule_dir.exists():
        print(f"❌ Molecule directory not found: {molecule_dir}")
        print(
            "   Run: python -m download.download_mol --source chembl --include-activities"
        )
        sys.exit(1)

    # Load data
    proteins_by_gene, proteins_by_uniprot = load_proteins(protein_dir)
    molecules = load_molecules(molecule_dir)
    targets = load_targets(molecule_dir)
    activities = load_activities(molecule_dir)

    if not proteins_by_uniprot:
        print("❌ No proteins loaded!")
        sys.exit(1)

    if not molecules:
        print("❌ No molecules loaded!")
        sys.exit(1)

    if not activities:
        print("❌ No activities loaded!")
        print(
            "   Run: python -m download.download_mol --source chembl --include-activities"
        )
        sys.exit(1)

    # Build interactions
    activity_types = set(args.activity_types)
    interactions = build_interactions(
        proteins_by_uniprot,
        molecules,
        targets,
        activities,
        min_pchembl=args.min_pchembl,
        activity_types=activity_types,
    )

    if not interactions:
        print("❌ No interactions could be built!")
        print("   Check that target UniProt IDs match your protein data")
        sys.exit(1)

    # Save interactions
    save_interactions(interactions, output_dir)

    print("\nDone!")
    print()
    print("Next steps:")
    print("  python -m prepare.prepare_p2m_data")
    print()


if __name__ == "__main__":
    main()
