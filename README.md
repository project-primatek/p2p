# Protein-Protein Interaction Prediction

AI-powered system to predict protein-protein interactions using AlphaFold structures and graph neural networks.

## 🎯 Project Goals

1. **Download AlphaFold Database** - Access structural data for millions of proteins
2. **Integrate Interaction Databases** - Use STRING, BioGRID, IntAct for training data
3. **Build GNN Model** - Graph neural network to predict protein binding
4. **Visualize Complexes** - 3D visualization of protein interactions

## 🏗️ Architecture

```
Input: Protein Structures (AlphaFold)
   ↓
Feature Extraction (sequence, structure, surface properties)
   ↓
Graph Neural Network (protein-protein interaction prediction)
   ↓
Output: Interaction probability + binding site prediction
```

## 📊 Datasets

- **AlphaFold DB**: 200M+ protein structures
- **STRING**: Known protein interactions (experimental + predicted)
- **BioGRID**: Curated protein interactions
- **PDB**: Experimentally solved protein complexes

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample data
python download_data.py --sample

# Train model
python train_ppi_model.py

# Predict interactions
python predict_interactions.py --protein1 P53 --protein2 MDM2
```

## 📁 Project Structure

```
project-g23/
├── data/
│   ├── alphafold/          # AlphaFold structures (.pdb files)
│   ├── string/             # STRING interaction database
│   └── biogrid/            # BioGRID interaction data
├── models/
│   ├── gnn_model.py        # Graph neural network
│   ├── feature_extractor.py
│   └── protein_encoder.py
├── utils/
│   ├── download_alphafold.py
│   ├── parse_pdb.py
│   └── visualization.py
├── notebooks/
│   └── exploration.ipynb
└── scripts/
    ├── train.py
    └── predict.py
```

## 🧠 Model Features

### Input Features (per protein)
- **Sequence**: Amino acid sequence
- **Structure**: 3D coordinates from AlphaFold
- **Surface**: Exposed residues, pockets
- **Biochemical**: Charge, hydrophobicity, polarity
- **Evolutionary**: Conservation scores

### GNN Architecture
- Node features: Residue-level properties
- Edge features: Spatial distances, bonds
- Graph pooling: Protein-level embedding
- Interaction predictor: MLP on concatenated embeddings

## 📈 Expected Results

- **Accuracy**: 80-90% on known interactions (STRING database)
- **Novel predictions**: Discover new protein interactions
- **Binding sites**: Predict interaction interfaces
- **Applications**: Drug target discovery, pathway analysis

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **BioPython**: Protein structure parsing
- **py3Dmol**: 3D visualization
- **pandas**: Data processing
- **numpy**: Numerical computations

## 🎓 Scientific Background

### Why Protein-Protein Interactions Matter
- **Drug Discovery**: 40% of drugs target protein interactions
- **Disease Understanding**: Many diseases caused by aberrant interactions
- **Systems Biology**: Understanding cellular networks

### AlphaFold Revolution
- Solved protein folding problem
- 200M+ structures now available
- Enables large-scale interaction prediction

## 📝 TODO

- [ ] Download AlphaFold structures for key proteins
- [ ] Parse STRING database
- [ ] Build initial GNN model
- [ ] Train on known interactions
- [ ] Validate on test set
- [ ] Build web interface for predictions
- [ ] Deploy model

## 🔬 Example Use Cases

### 1. Drug Target Discovery
```python
# Find proteins that interact with disease target
target = "P53"  # Tumor suppressor
candidates = predict_interactors(target)
# Output: MDM2, MDM4, etc. (known oncology targets)
```

### 2. Pathway Analysis
```python
# Map complete interaction network
proteins = ["BRCA1", "BRCA2", "TP53", "ATM"]
network = build_interaction_network(proteins)
visualize_network(network)
```

### 3. Binding Site Prediction
```python
# Where do proteins bind?
protein_a = load_structure("P53")
protein_b = load_structure("MDM2")
binding_site = predict_interface(protein_a, protein_b)
visualize_complex(protein_a, protein_b, binding_site)
```

## 📚 References

- **AlphaFold**: Jumper et al., Nature 2021
- **STRING**: Szklarczyk et al., Nucleic Acids Research 2023
- **GNN Review**: Zhou et al., AI Open 2020
- **PPI Prediction**: Zhang et al., Bioinformatics 2022

## 🤝 Contributing

This is a research project. Contributions welcome!

## 📄 License

MIT License - Use freely for research and commercial applications

---

**Status**: 🚧 Under active development

**Contact**: Research project for protein interaction prediction

**Last Updated**: November 2024