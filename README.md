# Cella Nova

**Cella Nova** is a comprehensive cell modeling platform that uses deep learning to predict biomolecular interactions. The platform combines protein language models, graph neural networks, and attention mechanisms to model the complex interaction networks within cells.

> **Note**: This platform uses **real experimental data only**. No synthetic or generated sequences are used for training.

## Features

### Protein-Protein Interaction (PPI) Prediction
Predict whether two proteins interact using:
- **ESM-2 protein language model** for sequence embeddings
- **Graph Neural Networks** for protein interaction network topology
- **Structure encoding** via AlphaFold contact maps
- **Siamese network architecture** for pairwise prediction

### Protein-DNA Interaction Prediction
Predict transcription factor binding and protein-DNA interactions using:
- **ESM-2** for protein sequence encoding
- **CNN + Bi-LSTM + Attention** for DNA sequence encoding
- **Cross-attention mechanism** for modeling protein-DNA interactions
- **Binding affinity prediction** alongside binary classification

### Protein-RNA Interaction Prediction
Predict RNA-binding protein (RBP) interactions using:
- **ESM-2** for protein sequence encoding with RBP-specific attention
- **Multi-scale CNN + Bi-LSTM** for RNA sequence encoding
- **Structure-aware attention** to capture RNA secondary structure
- **Binding site localization** via attention weights

### Protein-Small Molecule Interaction Prediction
Predict drug-target interactions and binding affinities using:
- **ESM-2** for protein sequence encoding with binding pocket attention
- **SMILES-based molecule encoder** with multi-scale CNN + Transformer
- **Pharmacophore attention** for molecule interpretability
- **Multi-task prediction**: binding, affinity, and interaction type

## Data Sources

All data comes from real experimental sources:

### Proteins
- **UniProt**: Protein sequences and annotations
- **AlphaFold**: Predicted 3D protein structures
- **STRING**: Protein-protein interaction networks

### DNA
- **ENCODE ChIP-seq**: Real transcription factor binding sites from experiments
- **JASPAR**: Experimentally validated TF binding motifs

### RNA
- **ENCODE eCLIP**: Real RNA-binding protein binding sites from experiments
- **ATtRACT**: Experimentally validated RNA-binding protein motifs
- **RNAcentral**: Real non-coding RNA sequences

### Small Molecules
- **ChEMBL**: Bioactivity data for drug-like molecules
- **PubChem**: Chemical compound structures

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cella-nova.git
cd cella-nova

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Real Data

The platform downloads real experimental data from public databases:

```bash
cd cella-nova

# Download protein data (UniProt + AlphaFold + STRING)
python -m download.download_pro --species "Homo sapiens" --include-structures --include-string

# Download DNA data (real TF binding sites)
python -m download.download_dna --source encode    # ENCODE ChIP-seq data
python -m download.download_dna --source jaspar    # JASPAR motifs
python -m download.download_dna --source all       # Both sources

# Download RNA data (real RBP binding sites)
python -m download.download_rna --source encode    # ENCODE eCLIP data
python -m download.download_rna --source attract   # ATtRACT motifs
python -m download.download_rna --source rnacentral # RNAcentral sequences
python -m download.download_rna --source all       # All sources

# Download molecule data (ChEMBL bioactivity data)
python -m download.download_mol --source chembl --include-activities
```

### 2. Prepare Data for Training

After downloading, prepare the data using the preparation scripts:

```bash
# Prepare all interaction types at once
python -m prepare.prepare_all

# Or prepare individual interaction types
python -m prepare.prepare_p2p_data  # Protein-Protein
python -m prepare.prepare_p2d_data  # Protein-DNA
python -m prepare.prepare_p2r_data  # Protein-RNA
python -m prepare.prepare_p2m_data  # Protein-Molecule

# With custom parameters
python -m prepare.prepare_all --negative-ratio 2.0 --seed 123
```

### 3. Train Models

```bash
# Train protein-protein interaction model
python -m model.model_p2p --data-dir data/prepared/p2p --epochs 50

# Train protein-DNA interaction model
python -m model.model_p2d --data-dir data/prepared/p2d --epochs 50

# Train protein-RNA interaction model
python -m model.model_p2r --data-dir data/prepared/p2r --epochs 50

# Train protein-molecule interaction model
python -m model.model_p2m --data-dir data/prepared/p2m --epochs 50
```

### 4. Make Predictions

```python
from model import PPIModel, ProteinDNAModel, ProteinRNAModel, ProteinMoleculeModel

# Load trained models
ppi_model = PPIModel.load("ppi_model.pt")
p2d_model = ProteinDNAModel.load("pdna_model.pt")
p2r_model = ProteinRNAModel.load("p2r_model.pt")
p2m_model = ProteinMoleculeModel.load("p2m_model.pt")

# Predict protein-protein interaction
score = ppi_model.predict(protein_a_seq, protein_b_seq)

# Predict protein-DNA binding
binding_prob, affinity = p2d_model.predict(protein_seq, dna_seq)

# Predict protein-RNA binding
binding_prob, affinity, binding_sites = p2r_model.predict(protein_seq, rna_seq)

# Predict drug-target interaction
result = p2m_model.predict(protein_seq, smiles)
```

## Model Performance

### PPI Model
| Metric | Score |
|--------|-------|
| AUC | 0.9999 |
| Precision | 0.99 |
| Recall | 1.00 |
| F1 Score | 0.995 |

### Protein-DNA Model (3 epochs)
| Metric | Score |
|--------|-------|
| AUC | 0.7087 |
| F1 Score | 0.5745 |
| Precision | 0.5870 |
| Recall | 0.5625 |

### Protein-RNA Model
*Training in progress*

### Protein-Molecule Model
*Training in progress*

## Project Structure

```
cella-nova/
├── data/
│   ├── proteins/           # Raw protein data (UniProt, AlphaFold, STRING)
│   ├── pdna/               # Raw protein-DNA data (ENCODE, JASPAR)
│   ├── rna/                # Raw RNA data (ENCODE eCLIP, ATtRACT, RNAcentral)
│   ├── molecules/          # Raw molecule data (ChEMBL)
│   └── prepared/           # Prepared training data
│       ├── p2p/            # Protein-Protein prepared data
│       ├── p2d/            # Protein-DNA prepared data
│       ├── p2r/            # Protein-RNA prepared data
│       └── p2m/            # Protein-Molecule prepared data
├── download/               # Data download scripts
│   ├── __init__.py
│   ├── download_pro.py     # Download protein data
│   ├── download_dna.py     # Download DNA data (ENCODE ChIP-seq, JASPAR)
│   ├── download_rna.py     # Download RNA data (ENCODE eCLIP, ATtRACT, RNAcentral)
│   └── download_mol.py     # Download molecule data (ChEMBL)
├── prepare/                # Data preparation scripts
│   ├── __init__.py
│   ├── prepare_all.py      # Master preparation script
│   ├── prepare_p2p_data.py # Protein-Protein preparation
│   ├── prepare_p2d_data.py # Protein-DNA preparation
│   ├── prepare_p2r_data.py # Protein-RNA preparation
│   └── prepare_p2m_data.py # Protein-Molecule preparation
├── model/                  # Neural network models
│   ├── __init__.py
│   ├── model_p2p.py        # Protein-protein interaction model
│   ├── model_p2d.py        # Protein-DNA interaction model
│   ├── model_p2r.py        # Protein-RNA interaction model
│   └── model_p2m.py        # Protein-small molecule interaction model
├── docs/                   # Documentation
│   └── DATA_PREPARATION.md
├── requirements.txt
└── README.md
```

## Download Scripts

### download/download_pro.py - Protein Data
Downloads protein sequences and structures from UniProt, AlphaFold, and STRING.

```bash
# All proteins for a species
python -m download.download_pro --species "Homo sapiens"

# With AlphaFold structures and STRING interactions
python -m download.download_pro --species "Homo sapiens" --include-structures --include-string

# Filtered by function
python -m download.download_pro --species "Homo sapiens" --filter dna-binding
python -m download.download_pro --species "Homo sapiens" --filter rna-binding
python -m download.download_pro --species "Homo sapiens" --filter kinase
```

### download/download_dna.py - DNA Data
Downloads real transcription factor binding data from ENCODE and JASPAR.

```bash
# Download ENCODE ChIP-seq binding sites (real experimental data)
python -m download.download_dna --source encode --max-experiments 100

# Download JASPAR motifs (experimentally validated)
python -m download.download_dna --source jaspar

# Download from all sources
python -m download.download_dna --source all
```

### download/download_rna.py - RNA Data
Downloads real RNA-binding protein data from ENCODE, ATtRACT, and RNAcentral.

```bash
# Download ENCODE eCLIP binding sites (real experimental data)
python -m download.download_rna --source encode --max-experiments 100

# Download ATtRACT motifs (experimentally validated)
python -m download.download_rna --source attract

# Download real RNA sequences from RNAcentral
python -m download.download_rna --source rnacentral --max-sequences 10000

# Download from all sources
python -m download.download_rna --source all
```

### download/download_mol.py - Molecule Data
Downloads small molecule data from ChEMBL.

```bash
# Download molecules
python -m download.download_mol --source chembl --max-molecules 10000

# Include bioactivity data
python -m download.download_mol --source chembl --include-activities --target-organism "Homo sapiens"
```

## Data Preparation Scripts

The prepare module processes raw downloaded data into training-ready datasets.

### prepare/prepare_all.py - Master Script
Runs all data preparations with consistent parameters.

```bash
# Prepare all interaction types
python -m prepare.prepare_all

# Prepare specific types only
python -m prepare.prepare_all --types p2p p2d

# Custom parameters
python -m prepare.prepare_all --negative-ratio 2.0 --train-ratio 0.7 --seed 123
```

### Individual Preparation Scripts
Each script prepares data for a specific interaction type:

```bash
python -m prepare.prepare_p2p_data  # Protein-Protein interactions
python -m prepare.prepare_p2d_data  # Protein-DNA interactions
python -m prepare.prepare_p2r_data  # Protein-RNA interactions
python -m prepare.prepare_p2m_data  # Protein-Molecule interactions
```

See `docs/DATA_PREPARATION.md` for detailed documentation.

## Architecture Overview

### PPI Model
```
Protein A Sequence ──► ESM-2 Encoder ──┐
                                       ├──► Cross-Modal Fusion ──► MLP ──► Interaction Score
Protein B Sequence ──► ESM-2 Encoder ──┤
                                       │
Network Topology ────► GNN Encoder ────┘
```

### Protein-DNA Model
```
Protein Sequence ──► ESM-2 Encoder ──────────────┐
                                                  ├──► Cross-Attention ──► MLP ──► Binding Prediction
DNA Sequence ──► CNN ──► Bi-LSTM ──► Attention ──┘
```

### Protein-RNA Model
```
Protein Sequence ──► ESM-2 + RBP Attention ──────────────────────┐
                                                                  ├──► Cross-Attention ──► MLP ──► Binding + Sites
RNA Sequence ──► Multi-scale CNN ──► Bi-LSTM ──► Structure Attn ─┘
```

### Protein-Molecule Model
```
Protein Sequence ──► ESM-2 + Pocket Attention ───────────────┐
                                                              ├──► Cross-Attention ──► Multi-task Head
SMILES ──► Multi-scale CNN ──► Transformer ──► Pharm Attn ───┘
                                                              │
                                              ┌───────────────┼───────────────┐
                                              ▼               ▼               ▼
                                          Binding        Affinity      Interaction Type
```

## Future Directions

The following interaction types are planned for future development:

- **RNA-RNA Interactions** - miRNA-mRNA targeting, lncRNA interactions
- **Post-Translational Modifications** - Phosphorylation, ubiquitination sites
- **Protein Localization** - Subcellular compartment prediction
- **3D Genome Interactions** - Enhancer-promoter loops, chromatin organization
- **Gene Regulatory Networks** - Transcription factor cascades

## References

### Databases
- [STRING Database](https://string-db.org/) - Protein-protein interaction networks
- [AlphaFold Database](https://alphafold.ebi.ac.uk/) - Protein structure predictions
- [JASPAR](https://jaspar.genereg.net/) - Transcription factor binding profiles
- [ENCODE](https://www.encodeproject.org/) - ChIP-seq and eCLIP experimental data
- [UniProt](https://www.uniprot.org/) - Protein sequence and annotation
- [ATtRACT](https://attract.cnic.es/) - RNA-binding protein motifs
- [RNAcentral](https://rnacentral.org/) - Non-coding RNA sequences
- [ChEMBL](https://www.ebi.ac.uk/chembl/) - Bioactivity database

### Key Papers
- Jumper et al. (2021) - "Highly accurate protein structure prediction with AlphaFold" - *Nature*
- Lin et al. (2023) - "Evolutionary-scale prediction of atomic-level protein structure with a language model" - *Science*
- Szklarczyk et al. (2023) - "The STRING database in 2023" - *Nucleic Acids Research*
- Van Nostrand et al. (2020) - "A large-scale binding and functional map of human RNA-binding proteins" - *Nature*

## License

MIT