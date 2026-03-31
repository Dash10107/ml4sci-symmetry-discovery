# Project Structure Guide

This document provides a detailed breakdown of the project organization and file purposes.

## Directory Layout

```
gsoc/
│
├── notebooks/                           # All Jupyter notebooks organized by task
│   │
│   ├── MNIST_Symmetry_Complete.ipynb  # 🔵 COMPLETE WORKFLOW
│   │                                    # Contains all tasks in a single notebook
│   │                                    # Best for understanding the full pipeline
│   │
│   ├── task1_vae/                      # 📁 TASK 1: VAE Training
│   │   └── Task_1_VAE.ipynb            # Train Variational Autoencoder
│   │                                    # Output: Learned 2D latent space
│   │                                    # Model: vae.pth (saved locally)
│   │
│   ├── task2_supervised_symmetry/      # 📁 TASK 2: Supervised Discovery
│   │   └── Task_2_Supervised_Symmetry.ipynb
│   │                                    # Learn rotation transformations with labels
│   │                                    # Output: latent_rotation_mlp.pth
│   │
│   ├── task3_unsupervised_symmetry/    # 📁 TASK 3: Unsupervised Discovery
│   │   └── Task_3_Unsupervised_Symmetry.ipynb
│   │                                    # Discover Lie group structure without labels
│   │                                    # Output: latent_transformation_G.pth
│   │
│   └── bonus_rotation_invariant/       # 📁 BONUS: Invariant Classifiers
│       └── Bonus_Rotation_Invariant.ipynb
│                                        # Build rotation-invariant classifiers
│                                        # Output: invariant_classifier.pth
│
├── dataset/                             # 📊 MNIST Data
│   ├── train.csv                        # 60,000 training samples (785 columns)
│   └── test.csv                         # 10,000 test samples
│
├── models/                              # 💾 Saved Model Checkpoints
│   ├── invariant_classifier.pth         # Group-pooled rotation invariant classifier
│   ├── latent_classifier.pth            # Standard latent space classifier
│   ├── latent_rotation_mlp.pth          # MLP for supervised rotation (Task 2)
│   └── latent_transformation_G.pth      # Lie group generator matrix (Task 3)
│
├── docs/                                # 📚 Documentation
│   ├── PROJECT_STRUCTURE.md             # This file
│   ├── THEORY.md                        # Mathematical background
│   └── RESULTS.md                       # Experimental results and metrics
│
└── README.md                            # 📖 Main project documentation

```

## Notebook Execution Order

### Option 1: Sequential Learning (Recommended for beginners)
1. `notebooks/task1_vae/Task_1_VAE.ipynb`
2. `notebooks/task2_supervised_symmetry/Task_2_Supervised_Symmetry.ipynb`
3. `notebooks/task3_unsupervised_symmetry/Task_3_Unsupervised_Symmetry.ipynb`
4. `notebooks/bonus_rotation_invariant/Bonus_Rotation_Invariant.ipynb`

### Option 2: All-in-One
- `notebooks/MNIST_Symmetry_Complete.ipynb` (contains all tasks)

## File Dependencies

```
Task 1 (VAE) → Task 2 (Supervised) → Task 3 (Unsupervised) → Bonus (Invariant)
     ↓              ↓                      ↓                       ↓
  vae.pth    rotation_mlp.pth      transformation_G.pth   invariant_classifier.pth
```

**Note**: Each task requires the VAE trained in Task 1. Task 2 and 3 are independent of each other.

## Model Files Explained

### `vae.pth` (Task 1 output)
- **What**: Trained Variational Autoencoder
- **Architecture**: 
  - Encoder: 784 → 400 → 2 (mean & logvar)
  - Decoder: 2 → 400 → 784
- **Used by**: All subsequent tasks

### `latent_rotation_mlp.pth` (Task 2 output)
- **What**: MLP that predicts latent transformations given rotation angles
- **Input**: (z, θ) - latent vector + rotation angle
- **Output**: z' - transformed latent vector
- **Training**: Supervised with known rotation angles

### `latent_transformation_G.pth` (Task 3 output)
- **What**: Lie algebra generator matrix A (2×2)
- **Purpose**: Generates rotation transformations via exp(tA)
- **Training**: Unsupervised, using reconstruction loss only
- **Key Property**: Encodes SO(2) group structure

### `latent_classifier.pth` (Bonus)
- **What**: Standard MLP classifier on latent space
- **Purpose**: Baseline for comparison
- **Performance**: Poor on rotated inputs

### `invariant_classifier.pth` (Bonus)
- **What**: Group-pooled rotation invariant classifier
- **Purpose**: Digit classification regardless of rotation
- **Method**: Averages over rotational orbit before classification
- **Performance**: Robust to arbitrary rotations

## Dataset Format

### train.csv & test.csv
```
Column 0: label (0-9)
Columns 1-784: pixel values (0-255, flattened 28×28 image)
```

### How images are loaded:
```python
image = row[1:].reshape(28, 28)
label = row[0]
```

## Adding Your Own Experiments

### To add a new task:
1. Create folder: `notebooks/task4_your_experiment/`
2. Add notebook: `Task_4_Your_Experiment.ipynb`
3. Document in this file
4. Update main README.md

### To save new models:
```python
torch.save(model.state_dict(), '../models/your_model_name.pth')
```

## Development Environment

### Recommended Setup:
- **Platform**: Google Colab (free GPU)
- **GPU**: T4 (16GB VRAM)
- **Runtime**: Python 3.10+
- **Key Libraries**: PyTorch, NumPy, Matplotlib

### Local Development:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib scipy opendatasets jupyter
```

## Version Control Notes

### Files to commit:
- ✅ Notebooks (`.ipynb`)
- ✅ Documentation (`.md`)
- ✅ README and structure files

### Files to ignore (add to `.gitignore`):
- ❌ Model checkpoints (`.pth`) - too large
- ❌ Dataset files (`.csv`) - downloadable
- ❌ Checkpoint files (`.ipynb_checkpoints/`)
- ❌ Python cache (`__pycache__/`, `*.pyc`)

### Recommended `.gitignore`:
```
# Model weights
models/*.pth
*.pth

# Dataset
dataset/*.csv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.pyc
*.pyo

# Environment
venv/
.env
```

## Questions?

- Check the main [README.md](../README.md) for project overview
- See [THEORY.md](THEORY.md) for mathematical explanations
- Review [RESULTS.md](RESULTS.md) for expected outcomes

---

**Maintained by**: Project Team | **Last Updated**: March 2026
