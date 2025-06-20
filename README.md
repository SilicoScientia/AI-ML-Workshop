# Molecular Docking Workshop with AutoDock Vina and Machine Learning

> Tip: Right-click the Colab badge and select "Open link in new tab" for best experience.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UhtkvglAFv1-NmEuX4XwiNIDB1EW2vmk#scrollTo=RXchmRP3Ad38)

A comprehensive workshop on molecular docking using AutoDock Vina integrated with machine learning techniques for binding energy prediction.

## Quick Start

### Download the Repository
```bash
# Option 1: Clone the repository
git clone https://github.com/SilicoScientia/AI-ML-Workshop.git

# Option 2: Download ZIP file
# Visit: https://github.com/SilicoScientia/AI-ML-Workshop/archive/refs/heads/main.zip
```

### Follow the Workshop
1. **Google Colab (Recommended)**: Open the [Colab Notebook](https://colab.research.google.com/drive/1UhtkvglAFv1-NmEuX4XwiNIDB1EW2vmk#scrollTo=RXchmRP3Ad38)
2. **Local Setup**: Follow the documentation in the repository


## Workshop Overview

This workshop covers:
- AutoDock Vina molecular docking
- Machine learning integration with Graph Convolutional Networks (GCN)
- ADMET property prediction
- Molecular visualization and analysis
- Binding energy prediction for new compounds

## Repository Structure

```
AI-ML-Workshop/
├── AutoDock_Vina/          # Traditional docking workflow
│   ├── vina.exe           # AutoDock Vina executable
│   ├── 3ue4.pdb           # Example protein receptor
│   ├── ligand.pdb         # Example ligand
│   └── Documentation files
├── ML/                    # Machine learning integration
│   ├── train_gcn.py       # GCN model training
│   ├── predict_gcn.py     # Binding energy prediction
│   ├── requirements.txt   # Python dependencies
│   ├── Protein/           # Protein files
│   ├── Ligands_docking/   # Training dataset (20 ligands)
│   └── Ligands_prediction/ # Prediction dataset (188 ligands)
└── README.md
```

## Documentation

- **AutoDock Vina Manual**: `AutoDock_Vina/Autodock_Vina_Manual.pdf`
- **Colab Documentation**: `ML/ML_Colab_Documentation.pdf`

## Installation and Setup

All installation instructions and setup procedures are provided in:
- The Colab notebook
- The documentation files
- The workshop manual

Follow the step-by-step instructions in these resources for complete setup and execution.

## Features

- Complete AutoDock Vina workflow
- GCN-based binding energy prediction
- ADMET property analysis
- Molecular visualization tools
- Batch processing capabilities
- Performance evaluation metrics

## Learning Objectives

By completing this workshop, you will learn:
1. Molecular docking principles and AutoDock Vina usage
2. Machine learning integration in drug discovery
3. Binding energy prediction using GCN models
4. ADMET property analysis and drug-likeness assessment
5. Molecular visualization and interaction analysis

## Contact

For questions or issues:
- GitHub Issues: [AI-ML-Workshop Issues](https://github.com/SilicoScientia/AI-ML-Workshop/issues)
- Follow the documentation and Colab notebook for support

---

*This workshop combines traditional molecular docking with modern machine learning techniques for drug discovery research.* 
