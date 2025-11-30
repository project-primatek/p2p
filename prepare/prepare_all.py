#!/usr/bin/env python3
"""
Master Data Preparation Script for Cella Nova

This script orchestrates the preparation of all interaction data types:
- P2P: Protein-Protein Interactions
- P2D: Protein-DNA Interactions
- P2R: Protein-RNA Interactions
- P2M: Protein-Molecule Interactions

It provides a unified interface to run all preparation steps with consistent
parameters and generates a comprehensive summary report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Determine project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_p2p_preparation(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    negative_ratio: float = 1.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Run Protein-Protein Interaction data preparation"""
    try:
        # Try relative import first (when used as package)
        try:
            from .prepare_p2p_data import PPIDataPreparer
        except ImportError:
            # Fall back to direct import (when run as script)
            import sys

            sys.path.insert(0, str(SCRIPT_DIR))
            from prepare_p2p_data import PPIDataPreparer  # type: ignore

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPARING P2P (PROTEIN-PROTEIN INTERACTION) DATA")
        logger.info("=" * 70)

        preparer = PPIDataPreparer(
            data_dir=data_dir,
            output_dir=output_dir / "p2p",
            seed=seed,
            negative_ratio=negative_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        result = preparer.prepare()
        return result

    except ImportError as e:
        logger.error(f"Failed to import P2P preparer: {e}")
        return None
    except Exception as e:
        logger.error(f"P2P preparation failed: {e}")
        return None


def run_p2d_preparation(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    negative_ratio: float = 1.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Run Protein-DNA Interaction data preparation"""
    try:
        # Try relative import first (when used as package)
        try:
            from .prepare_p2d_data import PDNADataPreparer
        except ImportError:
            # Fall back to direct import (when run as script)
            import sys

            if str(SCRIPT_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPT_DIR))
            from prepare_p2d_data import PDNADataPreparer  # type: ignore

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPARING P2D (PROTEIN-DNA INTERACTION) DATA")
        logger.info("=" * 70)

        preparer = PDNADataPreparer(
            data_dir=data_dir,
            output_dir=output_dir / "p2d",
            seed=seed,
            negative_ratio=negative_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        result = preparer.prepare()
        return result

    except ImportError as e:
        logger.error(f"Failed to import P2D preparer: {e}")
        return None
    except Exception as e:
        logger.error(f"P2D preparation failed: {e}")
        return None


def run_p2r_preparation(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    negative_ratio: float = 1.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Run Protein-RNA Interaction data preparation"""
    try:
        # Try relative import first (when used as package)
        try:
            from .prepare_p2r_data import PRNADataPreparer
        except ImportError:
            # Fall back to direct import (when run as script)
            import sys

            if str(SCRIPT_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPT_DIR))
            from prepare_p2r_data import PRNADataPreparer  # type: ignore

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPARING P2R (PROTEIN-RNA INTERACTION) DATA")
        logger.info("=" * 70)

        preparer = PRNADataPreparer(
            data_dir=data_dir,
            output_dir=output_dir / "p2r",
            seed=seed,
            negative_ratio=negative_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        result = preparer.prepare()
        return result

    except ImportError as e:
        logger.error(f"Failed to import P2R preparer: {e}")
        return None
    except Exception as e:
        logger.error(f"P2R preparation failed: {e}")
        return None


def run_p2m_preparation(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    negative_ratio: float = 1.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    affinity_threshold: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Run Protein-Molecule Interaction data preparation"""
    try:
        # Try relative import first (when used as package)
        try:
            from .prepare_p2m_data import PMolDataPreparer
        except ImportError:
            # Fall back to direct import (when run as script)
            import sys

            if str(SCRIPT_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPT_DIR))
            from prepare_p2m_data import PMolDataPreparer  # type: ignore

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPARING P2M (PROTEIN-MOLECULE INTERACTION) DATA")
        logger.info("=" * 70)

        preparer = PMolDataPreparer(
            data_dir=data_dir,
            output_dir=output_dir / "p2m",
            seed=seed,
            negative_ratio=negative_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            affinity_threshold=affinity_threshold,
        )

        result = preparer.prepare()
        return result

    except ImportError as e:
        logger.error(f"Failed to import P2M preparer: {e}")
        return None
    except Exception as e:
        logger.error(f"P2M preparation failed: {e}")
        return None


def generate_summary_report(
    results: Dict[str, Optional[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate a comprehensive summary report of all preparations"""

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_directory": str(output_dir),
        "preparation_results": {},
        "summary": {
            "total_interactions": 0,
            "total_proteins": 0,
            "successful_preparations": 0,
            "failed_preparations": 0,
        },
    }

    for prep_type, result in results.items():
        if result is not None:
            report["preparation_results"][prep_type] = {
                "status": "success",
                "num_interactions": result.get("num_interactions", 0),
                "train_size": result.get("train_size", 0),
                "val_size": result.get("val_size", 0),
                "test_size": result.get("test_size", 0),
                "output_dir": result.get("output_dir", ""),
            }
            report["summary"]["total_interactions"] += result.get("num_interactions", 0)
            report["summary"]["successful_preparations"] += 1

            # Add type-specific counts
            if "num_proteins" in result:
                report["summary"]["total_proteins"] += result.get("num_proteins", 0)
        else:
            report["preparation_results"][prep_type] = {
                "status": "failed",
                "error": "Preparation failed or module not available",
            }
            report["summary"]["failed_preparations"] += 1

    # Save report
    report_path = output_dir / "preparation_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPARATION SUMMARY REPORT")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Successful preparations: {report['summary']['successful_preparations']}"
    )
    logger.info(f"Failed preparations: {report['summary']['failed_preparations']}")
    logger.info(
        f"Total interactions prepared: {report['summary']['total_interactions']:,}"
    )
    logger.info(f"Total unique proteins: {report['summary']['total_proteins']:,}")
    logger.info("")

    for prep_type, prep_result in report["preparation_results"].items():
        status = prep_result["status"]
        if status == "success":
            logger.info(
                f"  {prep_type.upper()}: ✓ {prep_result['num_interactions']:,} interactions"
            )
            logger.info(
                f"       Train: {prep_result['train_size']:,} | Val: {prep_result['val_size']:,} | Test: {prep_result['test_size']:,}"
            )
        else:
            logger.info(f"  {prep_type.upper()}: ✗ Failed")

    logger.info("")
    logger.info(f"Full report saved to: {report_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Master Data Preparation Script for Cella Nova",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all interaction types
  python prepare_all.py --data-dir data --output-dir data/prepared

  # Prepare only P2P and P2D
  python prepare_all.py --types p2p p2d

  # Prepare with custom parameters
  python prepare_all.py --negative-ratio 2.0 --seed 123

  # Prepare with different split ratios
  python prepare_all.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing raw downloaded data (default: {DEFAULT_DATA_DIR})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save prepared data (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--types",
        nargs="+",
        choices=["p2p", "p2d", "p2r", "p2m", "all"],
        default=["all"],
        help="Interaction types to prepare (default: all)",
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples (default: 1.0)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for testing (default: 0.1)",
    )

    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=5.0,
        help="Minimum affinity (pIC50/pKd) for P2M positive interactions (default: 5.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue with other preparations if one fails",
    )

    args = parser.parse_args()

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.warning(f"Split ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which types to prepare
    types_to_prepare = set(args.types)
    if "all" in types_to_prepare:
        types_to_prepare = {"p2p", "p2d", "p2r", "p2m"}

    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " CELLA NOVA - DATA PREPARATION PIPELINE ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Interaction types: {', '.join(sorted(types_to_prepare))}")
    logger.info(f"Negative ratio: {args.negative_ratio}")
    logger.info(
        f"Split ratios: train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
    )
    logger.info(f"Random seed: {args.seed}")

    # Run preparations
    results: Dict[str, Optional[Dict[str, Any]]] = {}

    common_params = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "negative_ratio": args.negative_ratio,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    if "p2p" in types_to_prepare:
        results["p2p"] = run_p2p_preparation(**common_params)
        if results["p2p"] is None and not args.skip_errors:
            logger.error("P2P preparation failed. Use --skip-errors to continue.")
            sys.exit(1)

    if "p2d" in types_to_prepare:
        results["p2d"] = run_p2d_preparation(**common_params)
        if results["p2d"] is None and not args.skip_errors:
            logger.error("P2D preparation failed. Use --skip-errors to continue.")
            sys.exit(1)

    if "p2r" in types_to_prepare:
        results["p2r"] = run_p2r_preparation(**common_params)
        if results["p2r"] is None and not args.skip_errors:
            logger.error("P2R preparation failed. Use --skip-errors to continue.")
            sys.exit(1)

    if "p2m" in types_to_prepare:
        results["p2m"] = run_p2m_preparation(
            **common_params,
            affinity_threshold=args.affinity_threshold,
        )
        if results["p2m"] is None and not args.skip_errors:
            logger.error("P2M preparation failed. Use --skip-errors to continue.")
            sys.exit(1)

    # Generate summary report
    generate_summary_report(results, args.output_dir)

    # Final status
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)

    if successful == total:
        logger.info("✓ All data preparations completed successfully!")
        sys.exit(0)
    elif successful > 0:
        logger.warning(
            f"⚠ Completed {successful}/{total} preparations with some failures"
        )
        sys.exit(0 if args.skip_errors else 1)
    else:
        logger.error("✗ All data preparations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
