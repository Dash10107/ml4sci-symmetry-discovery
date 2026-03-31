# Setup and Installation Guide

Complete setup instructions for running the MNIST Symmetry Discovery project.

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 2GB free space

### Recommended Setup
- **GPU**: NVIDIA GPU with CUDA support (e.g., Google Colab T4)
- **RAM**: 16GB
- **Python**: 3.10+

---

## 🚀 Quick Setup (5 Minutes)

### Option 1: Google Colab (Recommended - No Installation!)

1. **Open Colab**: Go to [Google Colab](https://colab.research.google.com/)

2. **Upload Notebook**:
   - Click "File" → "Upload notebook"
   - Choose `notebooks/MNIST_Symmetry_Complete.ipynb`

3. **Enable GPU**:
   - Click "Runtime" → "Change runtime type"
   - Select "T4 GPU" under Hardware accelerator
   - Click "Save"

4. **Run All Cells**:
   - Click "Runtime" → "Run all"
   - Wait ~35 minutes for all tasks to complete

**That's it!** No local installation needed.

---

### Option 2: Local Installation

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gsoc-symmetry-discovery.git
cd gsoc-symmetry-discovery
```

#### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Expected installation time**: 3-5 minutes

#### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.0.0+cu118
CUDA available: True  # (or False if no GPU)
```

#### Step 5: Download Dataset

The dataset should already be in `dataset/` folder. If not:

**Option A - Manual Download:**
1. Visit [Kaggle MNIST CSV Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
2. Download `train.csv` and `test.csv`
3. Place in `dataset/` folder

**Option B - Using opendatasets:**
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
```

#### Step 6: Launch Jupyter

```bash
jupyter notebook
```

Navigate to `notebooks/` and open any notebook!

---

## 📦 Detailed Package Information

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.0+ | Deep learning framework |
| `torchvision` | 0.15+ | Vision utilities |
| `numpy` | 1.23+ | Numerical computations |
| `scipy` | 1.10+ | Matrix exponentials (Lie groups) |
| `matplotlib` | 3.6+ | Visualizations |
| `pandas` | 1.5+ | Data handling |
| `jupyter` | 1.0+ | Interactive notebooks |

### Installation Issues?

**Problem**: `pip install torch` takes forever

**Solution**: Use CPU-only version for testing:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: CUDA version mismatch

**Solution**: Check your CUDA version and install matching PyTorch:
```bash
nvcc --version  # Check CUDA version
# Then install appropriate version from pytorch.org
```

**Problem**: Import errors

**Solution**: Ensure virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

---

## 🎯 Running the Notebooks

### Sequential Execution (Recommended for Learning)

Run in this order to understand the progression:

```bash
# 1. Train VAE (Foundation)
jupyter notebook notebooks/task1_vae/Task_1_VAE.ipynb

# 2. Supervised Discovery
jupyter notebook notebooks/task2_supervised_symmetry/Task_2_Supervised_Symmetry.ipynb

# 3. Unsupervised Discovery (Key contribution)
jupyter notebook notebooks/task3_unsupervised_symmetry/Task_3_Unsupervised_Symmetry.ipynb

# 4. Rotation Invariance (Application)
jupyter notebook notebooks/bonus_rotation_invariant/Bonus_Rotation_Invariant.ipynb
```

**Total runtime**: ~35 minutes on GPU, ~2 hours on CPU

### All-in-One Execution

```bash
jupyter notebook notebooks/MNIST_Symmetry_Complete.ipynb
```

**Runtime**: ~40 minutes on GPU

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Out of Memory" Error

**Symptoms**: CUDA out of memory during training

**Solutions**:
```python
# In notebook, reduce batch size
batch_size = 64  # Try 32 or 16

# Or use CPU
device = torch.device('cpu')
```

#### 2. Notebook Kernel Crashes

**Symptoms**: Kernel dies during execution

**Solutions**:
- Restart kernel and run cells individually
- Clear output: Cell → All Output → Clear
- Use Google Colab instead

#### 3. Slow Training

**Symptoms**: Training takes hours

**Solutions**:
- Enable GPU in Colab (Runtime → Change runtime type)
- Reduce epochs for testing:
  ```python
  num_epochs = 10  # Instead of 50
  ```

#### 4. Missing Dataset

**Symptoms**: `FileNotFoundError: dataset/train.csv`

**Solutions**:
```bash
# Check current directory
pwd  # Should be in gsoc/ root

# Verify dataset exists
ls dataset/

# If missing, download from Kaggle
```

#### 5. Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'torch'`

**Solutions**:
```bash
# Verify virtual environment is active
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt

# Or install individually
pip install torch torchvision numpy matplotlib
```

---

## 🔧 Advanced Configuration

### Using Different Python Versions

```bash
# Specify Python version
python3.10 -m venv venv
```

### Installing Development Dependencies

```bash
pip install jupyter black flake8 pytest
```

### Running on Specific GPU

```python
import torch

# Specify GPU device
device = torch.device('cuda:0')  # First GPU
# device = torch.device('cuda:1')  # Second GPU

model = model.to(device)
```

### Saving Computation Time

For quick testing, modify notebooks:

```python
# Reduce dataset size
train_size = 1000  # Instead of full 60,000

# Reduce epochs
num_epochs = 5  # Instead of 50

# Reduce rotation angles
rotation_angles = [0, 90, 180, 270]  # Instead of 30° steps
```

---

## 📊 Expected Outputs

After successful setup and execution, you should see:

### Files Created:
```
models/
  ├── vae.pth (or checkpoint from Task 1)
  ├── latent_rotation_mlp.pth
  ├── latent_transformation_G.pth
  ├── latent_classifier.pth
  └── invariant_classifier.pth

output/
  ├── classifier_accuracy_comparison.png
  ├── learned_generator_A_heatmap.png
  ├── lie_group_rotation_digit_1.png
  ├── original_vs_reconstructed_images.png
  ├── tsne_latent_space.png
  └── tsne_original_vs_pooled.png
```

### Console Output:
```
Task 1: VAE training...
Epoch 50/50, Loss: 142.3
✅ VAE trained successfully

Task 2: Supervised discovery...
Rotation MSE: 0.08
✅ Supervised symmetry discovered

Task 3: Unsupervised discovery...
Group closure error: 0.023
✅ SO(2) symmetry discovered

Bonus: Rotation invariance...
Invariant accuracy: 99.4%
✅ Rotation-invariant classifier built
```

---

## 🎓 Learning Resources

### If You're New to PyTorch
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Fast.ai Deep Learning Course](https://course.fast.ai/)

### If You're New to Jupyter
- [Jupyter Notebook Tutorial](https://jupyter.org/try)
- Run cells with `Shift+Enter`
- Add cells with `B` (below) or `A` (above)

### If You're New to VAEs
- Read `docs/THEORY.md` for mathematical background
- [Tutorial on VAEs](https://arxiv.org/abs/1906.02691)

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Python version ≥ 3.8
- [ ] All packages installed (check with `pip list`)
- [ ] GPU available (if using CUDA)
- [ ] Dataset files in `dataset/`
- [ ] Jupyter starts without errors
- [ ] Can open and run a test notebook cell

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check documentation**: Review `docs/` folder
2. **Search error message**: Google the exact error
3. **GitHub Issues**: Check if others had same problem
4. **Ask mentors**: Email [ml4-sci@cern.ch](mailto:ml4-sci@cern.ch)

**When asking for help, include**:
- Operating system and Python version
- Full error message
- What you were trying to do
- What you've already tried

---

## 🎉 You're All Set!

If you've completed setup successfully:

✅ **Next Step**: Open `notebooks/MNIST_Symmetry_Complete.ipynb`  
✅ **Expected Time**: 40 minutes to run everything  
✅ **Expected Output**: All results in `output/` folder  

**Happy experimenting!** 🚀

---

**Last Updated**: March 2026  
**Tested On**: Python 3.10, PyTorch 2.0, CUDA 11.8
