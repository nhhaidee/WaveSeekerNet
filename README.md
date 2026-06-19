# WaveSeekerNet: Accurate Prediction of Influenza A Virus Subtypes and Host Source Using Attention-Based Deep Learning


## Introduction

Influenza A virus (IAV) poses a significant threat to animal health globally, with its ability to overcome species barriers and cause pandemics. Rapid and accurate IAV subtype and host source prediction are crucial for effective surveillance and pandemic preparedness. Deep learning has emerged as a powerful tool for analyzing viral genomic sequences, offering new ways to uncover hidden patterns associated with viral characteristics and host adaptation.

We introduce WaveSeekerNet, a novel deep learning model for accurate and rapid prediction of IAV subtypes and host source. The model leverages attention-based mechanisms and efficient token mixing schemes, including the Fourier Transform and the Wavelet Transform, to capture intricate patterns within viral RNA and protein sequences. Extensive experiments on diverse datasets demonstrate WaveSeekerNet’s superior performance to existing models that use the traditional self-attention mechanism. Notably, WaveSeekerNet rivals VADR (Viral Annotation DefineR) in subtype prediction using the high-quality RNA sequences, achieving the maximum score of 1.0 on metrics including the Balanced Accuracy, F1-score (Macro Average), and Matthews Correlation Coefficient (MCC). Our approach to subtype and host source prediction also exceeds the pre-trained ESM-2 (Evolutionary Scale Modeling) models with respect to generalization performance and computational cost. Furthermore, WaveSeekerNet exhibits remarkable accuracy in distinguishing between human, avian, and other mammalian hosts. The ability of WaveSeekerNet to flag potential cross-species transmission events underscores its significant value for real-time surveillance and proactive pandemic preparedness efforts. 

WaveSeekerNet’s superior performance, efficiency, and ability to flag potential cross-species transmission events highlight its potential for real-time surveillance and pandemic preparedness. This model represents a significant advancement in applying deep learning for IAV classification and holds promise for future epidemiological, veterinary studies, and public health interventions. 

This repository contains the source code and data used to train WaveSeekerNet. The preprint of WaveSeekerNet is now available on bioRxiv (https://www.biorxiv.org/content/10.1101/2025.02.25.639900v2) and under review with GigaScience.
## Installation

### Standard Installation
To install `WaveSeekerNet` and its required packages, clone the repository and run:

```bash
git clone https://github.com/nhhaidee/WaveSeekerNet.git
cd WaveSeekerNet
pip install .
```

For development work (including tests, linting, and building):

```bash
pip install -e .[dev]
```

### Conda Installation
An `environment.yml` file is provided to easily create a Conda environment with all PyTorch, CUDA, and package dependencies pre-configured:

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate waveseekernet
```

### Docker Containerization
Dockerfiles are provided to run WaveSeekerNet inside containers with full GPU or CPU support.

#### Option 1: GPU-Enabled Container (CUDA 12.1 + PyTorch)
Build the image:
```bash
docker build -t waveseekernet:latest .
```

Run a python script with GPU support:
```bash
docker run --gpus all -it --rm waveseekernet:latest python your_script.py
```

#### Option 2: CPU-Only Container
Build the image:
```bash
docker build -f Dockerfile.cpu -t waveseekernet:cpu .
```

Run container:
```bash
docker run -it --rm waveseekernet:cpu python your_script.py
```

### Singularity / Apptainer (for HPC Clusters)
For High-Performance Computing (HPC) clusters where root privileges are restricted, you can build a Singularity/Apptainer image.

#### Option 1: Build from Docker Hub (Easiest)
If you have pushed your Docker image to Docker Hub, you can pull and build it directly:
```bash
singularity build waveseekernet.sif docker://username/waveseekernet:latest
```

#### Option 2: Build from the Definition File
Build the Singularity Image File (`.sif`) locally using the recipe file:
```bash
singularity build waveseekernet.sif waveseekernet.def
```

#### Running the Singularity Container
Run a python script utilizing the GPU/CUDA context:
```bash
singularity run --nv waveseekernet.sif your_script.py
```

## Requirements

WaveSeekerNet requires Python 3.10+ and the following core dependencies:

1. **PyTorch** >= 2.4.1 (with GPU support highly recommended)
2. **[Pytorch Wavelet package]** >= 1.3.0
3. **[Pytorch Optimizer]** >= 3.1.1
4. Other dependencies: `scikit-learn>=1.5.1`, `numpy>=1.26.4`, `torchinfo==1.8.0`, `shap==0.48.0`, `PyWavelets`, `biopython>=1.80`, `complexcgr`, `seaborn`, `matplotlib`, `pyfastx`, `pandas`, `baycomp`.


## Data and Source Code

1. **Dataset Metadata**: Metadata for the datasets used in the paper can be found in GigaDB (https://doi.org/10.5524/102732).
2. **Sequence Data**: IAV HA and NA RNA/Protein sequences can be downloaded from EpiFlu GISAID database (https://www.gisaid.org/).
3. **Repository Structure**:
    - `src/WaveSeekerNet`: Contains code for the `WaveSeekerNet` model, blocks, classification head, and submodules.
    - `sampling.py`: Backward-compatibility wrapper for resampling functions.
    - `WaveSeekerNet_Demo.ipynb`: A complete, executable Jupyter notebook demonstration.

## Model Architecture Overview

WaveSeekerNet introduces an attention-based deep learning architecture designed for biological sequences, specifically optimizing feature representation across multi-scale dimensions.

### 1. Patch Extraction (`MakePatches`)
Biological sequences are represented in 2-D (Frequency Chaos Game Representation (FCGR) for DNA/RNA or residue-by-channel matrix for protein). These are divided into non-overlapping patches and mapped to an embedding space:
* **`patch_mode="patch"`**: Traditional non-overlapping patches.
* **`patch_mode="compress"`**: Compresses sequence dimensions before patch extraction.
* **`patch_mode="full"`**: Retains complete spatial dimensions using average pooling.

### 2. WaveSeekerBlock (Encoder)
Instead of traditional self-attention which suffers from high computational complexity, each block processes tokens in parallel:
* **Wavelet Head (`WaveNETHead`)**: Applies a 1-level Discrete Wavelet Transform (DWT), processes approximation and detail coefficients separately using a `StarLayer`, soft-thresholds noise using shrinkage regularization, and reconstructs via Inverse DWT.
* **Fourier Head (`FNETHead`)**: Applies a 2-D Real FFT, applies efficient multi-head linear self-attention, prunes low-magnitude frequencies using shrinkage, and reconstructs via Inverse FFT.
* **gMLP Head (`gMLPBlock`)**: Applies a spatial gating unit over token projections.
* **Merging (`StarLayer`)**: Integrates parallel heads.
* **Channel-Mixing (`SparseMoE` / `WaveExpert`)**: Processes the hidden dimension using a Sparse Mixture-of-Experts (SMoE) router selecting top-3 experts, or falls back to a single `WaveExpert`.

### 3. Classification Head (`ClassificationHead`)
Pools patch tokens using Global Expectation Pooling and feeds them to the classification head, which can utilize **Kolmogorov-Arnold Network (KAN)** layers (`KANLinear`) in place of standard fully-connected layers.

## Model Parameters Reference

The `WaveSeekerClassifier` class implements the scikit-learn estimator interface. Below is the list of initialization parameters:

| Parameter | Type | Default      | Description |
| :--- | :--- |:-------------| :--- |
| `seq_L` | `int` | *Required*   | Sequence-length dimension of the input matrix. |
| `res_L` | `int` | *Required*   | Residue/feature-length dimension of the input matrix (e.g. 5 or 6 for one-hot, 21 for protein). |
| `n_channels` | `int` | *Required*   | Number of input channels (e.g., 1). |
| `patch_size` | `tuple[int, int]` | *Required*   | `(height, width)` size of each patch. |
| `n_out` | `int` | *Required*   | Number of output classes (subtypes/hosts). |
| `emb_dim` | `int` | `196`        | Dimensionality of the patch embeddings. |
| `wavelet_names` | `list[str] \| None` | `None`       | List of wavelet filter names (defaults to `["bior3.3", "sym4"]`). |
| `wave_dropout` | `float` | `0.5`        | Dropout rate inside WaveSeekerBlocks. |
| `use_fft` | `bool` | `True`       | Include Fourier (FNet) token-mixing head. |
| `use_wavelets` | `bool` | `True`       | Include wavelet token-mixing heads. |
| `use_gmlp` | `bool` | `True`       | Include gMLP token-mixing head. |
| `use_smoe` | `bool` | `True`       | Use Sparse Mixture-of-Experts inside encoder blocks. |
| `use_kan` | `bool` | `True`       | Use Kolmogorov-Arnold Network (KAN) layers in the classifier head. |
| `patch_mode` | `str` | `"compress"` | Patch extraction mode (`"patch"`, `"compress"`, or `"full"`). |
| `n_blocks` | `int` | `1`          | Number of stacked encoder blocks. |
| `final_dropout` | `float` | `0.5`        | Dropout rate in classification head. |
| `final_hidden_size`| `int` | `32`         | Hidden layer size of the classification head. |
| `epochs` | `int` | `30`         | Number of training epochs. |
| `batch_size` | `int` | `64`         | Training batch size. |
| `lr` | `float` | `1e-3`       | Initial learning rate. |
| `wd` | `float` | `0.0`        | Weight decay rate. |
| `optimizer_name` | `str` | `"Adan"`     | PyTorch optimizer name (supported by `pytorch_optimizer`). |
| `use_gc` | `bool` | `True`       | Use Gradient Centralization. |
| `use_lookahead` | `bool` | `True`       | Wrap optimizer with Lookahead. |
| `activation` | `Type[nn.Module]` | `ErMish`     | Activation function class. |
| `return_probs` | `bool` | `True`       | Whether to return softmax class probabilities. |
| `device` | `str \| torch.device \| None` | `None`       | Force device mapping (defaults to CUDA if available, else CPU). |

## How to Train WaveSeekerNet

Here is a quick-start guide to training the classifier:

### 1. Load Data
```python
import numpy as np

# Load preprocessed arrays (FCGR or one-hot)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')
```

### 2. Configure and Fit the Classifier
```python
from WaveSeekerNet import WaveSeekerClassifier

# Initialize the classifier
clf = WaveSeekerClassifier(
    n_channels=1,
    seq_L=64,                # e.g., 64 for k=6 FCGR (2**k)
    res_L=64,                # e.g., 64 for k=6 FCGR
    patch_size=(4, 4),
    n_out=len(np.unique(y_train)),
    batch_size=256,
    emb_dim=64,
    final_hidden_size=24,
    epochs=35,
    patch_mode="patch",
    wavelet_names=["sym4"],
    n_blocks=1,
    lr=0.0025
)

# Fit model (optionally provides validation data)
clf.fit(X_train, y_train)

# Predict labels and evaluate
y_pred = clf.predict(X_test)
```

### 3. Evaluate Results
```python
from sklearn.metrics import classification_report, balanced_accuracy_score, matthews_corrcoef

print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Sequence Preprocessing Utilities

The package provides memory-efficient utilities under `WaveSeekerNet.utils` to preprocess raw FASTA genomic sequences into One-Hot Encoded representation or Frequency Chaos Game Representation (FCGR). Both tools support **on-disk memory-mapping (`np.memmap`)** for handling very large datasets (500K+ sequences) without running out of RAM.

### 1. One-Hot Encoding DNA/RNA Sequences
Convert DNA sequences into a 3D one-hot encoded matrix. Non-standard bases (e.g., IUPAC codes, gaps, or `N`) are grouped under the ambiguous channel.

```python
from WaveSeekerNet.utils import fasta_to_one_hot

# Convert sequences from a FASTA file and save directly to disk
X_train, headers = fasta_to_one_hot(
    fasta_path="path/to/sequences.fasta",
    seq_len=2400,                   # Target sequence length (pads or truncates)
    res_l=5,                        # 5 channels: A->0, C->1, G->2, T/U->3, Ambiguous/Padding->4
    convert_ambiguous_to_n=True,    # Map all non-ACGT bases to N (index 4)
    chunk_size=50000,               # Process in batches to limit RAM usage
    out_filename="X_train_onehot.npy" # Creates a disk-backed memory-mapped array
)

print(X_train.shape) # (n_sequences, 5, 2400)
```

### 2. Frequency Chaos Game Representation (FCGR)
Convert DNA sequences to FCGR frequency matrices using the `complexCGR` package. Non-standard characters are automatically cleaned and mapped to `N` before CGR generation.

```python
from WaveSeekerNet.utils import fasta_to_fcgr

# Convert sequences to standardized FCGR matrices and save to disk
X_train, headers = fasta_to_fcgr(
    fasta_path="path/to/sequences.fasta",
    k=6,                            # k-mer size (generates a 64x64 image)
    standardize=True,               # Normalize values to be independent of sequence length
    chunk_size=10000,               # Process in batches to limit RAM usage
    out_filename="X_train_fcgr.npy" # Creates a disk-backed memory-mapped array
)

print(X_train.shape) # (n_sequences, 64, 64)
```

### 3. Loading Large Datasets for Training (Zero-RAM Overhead)
When training with large datasets (e.g., 500K sequences), load the generated files using NumPy's `mmap_mode='r'`. This ensures that batches are read dynamically from the disk during training rather than filling up your RAM:

```python
import numpy as np
import torch

# Load in read-only memmap mode (takes virtually 0 RAM)
X_train_mmap = np.load("X_train_onehot.npy", mmap_mode="r")
y_train = np.load("y_train.npy")

# Convert to PyTorch tensors (shares memory buffer, no copying)
X_train_tensor = torch.from_numpy(X_train_mmap)
y_train_tensor = torch.from_numpy(y_train)

# Pass to DataLoader
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
```

## Advanced Features

### 1. Model Structure and Parameter Count
You can print and inspect the architecture of the underlying PyTorch model using the `summary()` method:

```python
# Displays layer shapes, parameter counts, and MACs
summary_str = clf.summary()
print(summary_str)
```

### 2. Model Explainability (SHAP Support)
`WaveSeekerClassifier` integrates with the `shap` package to generate feature attribution maps for biological sequences. To avoid GPU out-of-memory errors on large sequence sizes, `explain()` partitions evaluation into mini-batches:

```python
# Compute SHAP values for explaining predictions
shap_values = clf.explain(
    X_explain=X_test[:100],            # Sequence samples to explain
    background_data=X_train[:50],      # Optional: representative baseline data
    explainer_type="gradient",         # "gradient" (Integrated Gradients), "deep" (DeepLIFT), or "kernel"
    output_type="logits",              # Explain "logits" (recommended) or "probs"
    batch_size=32                      # Batch size used during SHAP evaluation
)

# Output shape matches inputs stacked with class channels:
# Single-channel: (n_samples, res_L, seq_L, n_classes)
# Multi-channel:  (n_samples, n_channels, res_L, seq_L, n_classes)
print("SHAP values shape:", shap_values.shape)
```

### 3. Dataset Resampling (Class Imbalance)
To combat highly skewed classification targets (e.g., highly prevalent vs. rare viral subtypes/hosts), the package includes helper resampling utilities under `WaveSeekerNet` (or `WaveSeekerNet.utils`):

```python
from WaveSeekerNet import resampling, get_rare_sequence

# Downsamples over-represented classes and upsamples under-represented classes
X_resampled, y_resampled = resampling(
    X_train, 
    y_train, 
    n_downsamples=16000, 
    n_upsamples=600
)

# Specifically oversamples extremely rare categories (classes with < s_splits samples)
X_rare, y_rare, X_other, y_other = get_rare_sequence(
    X_train, 
    y_train, 
    s_splits=10, 
    n_samples=600
)
```

## Jupyter Notebook Demo

A fully functional demonstration showing data loading, resampling, model training, evaluation, and explanation can be found in [WaveSeekerNet_Demo.ipynb](WaveSeekerNet_Demo.ipynb).

## Contributors and Maintainers

* [Hai Nguyen](https://github.com/nhhaidee) ([CFIA-NCFAD](https://github.com/CFIA-NCFAD), Department of Computer Science, University of Manitoba) - designed the models, wrote the code/manuscript, prepared data, trained models, performed experiments and completed the data analysis.
* [Josip Rudar](https://github.com/jrudar) ([CFIA-NCFAD](https://github.com/CFIA-NCFAD), Department of Integrative Biology & Centre for Biodiversity Genomics, University of Guelph) - designed models, wrote the code, reviewed/edited the manuscript, provided guidance on the project, and provided feedback on the experiments.


[Pytorch Wavelet package]: https://github.com/fbcotter/pytorch_wavelets
[Pytorch Optimizer]: https://github.com/kozistr/pytorch_optimizer