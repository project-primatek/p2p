# Protein-Protein Interaction Prediction

Predict protein-protein interactions using AlphaFold structures and machine learning.

## Goal

Build a machine learning model that predicts whether two proteins interact, using:
- **3D protein structures** from AlphaFold as input features
- **Known interactions** from STRING database as training labels

## Datasets

### AlphaFold Database
- Source: https://alphafold.ebi.ac.uk/
- Contains: Predicted 3D structures for 200M+ proteins
- Format: PDB files with atomic coordinates
- Use: Extract structural features (surface properties, binding sites, shape)

### STRING Database
- Source: https://string-db.org/
- Contains: Known and predicted protein-protein interactions
- Format: Protein pairs with confidence scores (0-1000)
- Use: Training labels (interacting vs non-interacting pairs)

## Download Data

### 1. Download Proteome + AlphaFold Structures

```bash
# Download all proteins and structures for a species
python download_proteome.py --species "Mycoplasma genitalium"

# For larger organisms, use more threads
python download_proteome.py --species "Homo sapiens" --threads 20

# Or use UniProt proteome ID directly
python download_proteome.py --uniprot-id UP000000807
```

Output:
- `data/{species}/proteins.fasta` - All protein sequences
- `data/{species}/structures/` - AlphaFold PDB files

### 2. Download STRING Interactions

```bash
# Download interactions for a species
python download_string.py --species "Mycoplasma genitalium"

# Filter by confidence score (700+ = high confidence)
python download_string.py --taxon-id 9606 --score 700
```

Output:
- `data/{species}/string/string_interactions.tsv` - Interaction pairs with scores

## ML Approach

### Data Preparation

1. **Positive samples**: Protein pairs from STRING with high confidence scores (≥700)
2. **Negative samples**: Random protein pairs not in STRING (or low confidence <150)
3. **Train/test split**: 80/20, ensuring no protein leakage between sets

### Feature Extraction (per protein)

From AlphaFold PDB structures:
- **Geometric**: Surface area, volume, radius of gyration
- **Chemical**: Charge distribution, hydrophobicity patches
- **Structural**: Secondary structure composition, exposed residues
- **Sequence**: Amino acid frequencies, length

For protein pairs:
- Concatenate or compare individual features
- Surface complementarity scores
- Docking-based features (optional)

### Model Architecture

**Option 1: Classical ML**
```
Protein A features + Protein B features → Random Forest / XGBoost → Interaction probability
```

**Option 2: Graph Neural Network**
```
Protein A graph + Protein B graph → GNN encoder → Pair embedding → MLP → Interaction probability
```

Where each protein is a graph:
- Nodes = amino acid residues
- Edges = spatial contacts (Cα distance < 8Å)
- Node features = residue properties

### Training

```python
# Pseudocode
for protein_a, protein_b, label in dataloader:
    feat_a = extract_features(protein_a)
    feat_b = extract_features(protein_b)
    pred = model(feat_a, feat_b)
    loss = binary_cross_entropy(pred, label)
    loss.backward()
```

### Evaluation Metrics
- AUROC, AUPRC (handles class imbalance)
- Precision, Recall, F1 at various thresholds

## Similar Projects

### GitHub Repositories
- [DeepPPI](https://github.com/hashemifar/DeepPPI) - CNN-based PPI prediction from sequence
- [PIPR](https://github.com/muhaochen/seq_ppi) - Siamese RNN for PPI prediction
- [GNN-PPI](https://github.com/lvguofeng/GNN_PPI) - Graph neural networks for PPI
- [D-SCRIPT](https://github.com/samsledje/D-SCRIPT) - Structure-aware deep learning for PPI
- [AlphaFold-Multimer](https://github.com/deepmind/alphafold) - Direct complex structure prediction

### Key Papers
- Jumper et al. (2021) - "Highly accurate protein structure prediction with AlphaFold" - *Nature*
- Szklarczyk et al. (2023) - "The STRING database in 2023" - *Nucleic Acids Research*
- Evans et al. (2022) - "Protein complex prediction with AlphaFold-Multimer" - *bioRxiv*
- Sledzieski et al. (2021) - "D-SCRIPT: Predicting direct physical interactions" - *Cell Systems*
- Gainza et al. (2020) - "Deciphering interaction fingerprints from protein molecular surfaces" - *Nature Methods*

## Project Structure

```
p2p/
├── data/
│   └── {species}/
│       ├── proteins.fasta
│       ├── sequences/
│       ├── structures/
│       └── string/
├── download_proteome.py
├── download_string.py
└── requirements.txt
```

## License

MIT