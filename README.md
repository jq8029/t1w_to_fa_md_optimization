## A Systematic Evaluation of Deep Learning Strategies for T1-to-dMRI Prediction
This project investigates voxelwise prediction of **fractional anisotropy (FA)** and **mean diffusivity (MD)** from T1-weighted MRI using deep learning. The goal is to assess how **model architecture, spatial context, prediction strategy, and loss design** affect T1-to-dMRI prediction.

## Methods
We compared three architectures:

- **CNN**
- **U-Net**
- **TransUNet**

We also evaluated:

- Input patch sizes of **64<sup>3</sup>** and **80<sup>3</sup>**
- **Single-head** and **two-head** prediction
- Baseline reconstruction loss and an **FA-oriented SSIM loss**

All models predict voxelwise FA and MD maps from T1-weighted MRI.

## Key Results
- **U-Net and TransUNet** outperformed CNN, with FA MAE reduced by approximately 19%.
- Increasing the input patch size from **64<sup>3</sup> to 80<sup>3</sup>** improved both FA and MD prediction.
- **TransUNet** showed only marginal gains over U-Net.
- The **two-head prediction** strategy improved FA prediction while maintaining comparable MD performance.
- The **FA-oriented SSIM loss** improved the structural quality of FA predictions.

Overall, the results suggest that **spatial context and task-specific optimization are more important than increasing model complexity alone**.

## Experimental Design
Each experiment changes one factor while keeping the other settings fixed:

- **Network architecture**
- **Spatial context**
- **Prediction strategy**
- **Loss function**

### 1. Network Architecture
The three models provide different levels of spatial feature extraction:
- CNN: convolutional baseline
- U-Net: encoder-decoder with skip connections
- TransUNet: CNN-Transformer hybrid

The comparison evaluates whether increased architectural complexity improves prediction performance.

### 2. Input Patch Size
Larger patches provide more spatial context for voxelwise prediction. We compare:

- 64<sup>3</sup>
- 80<sup>3</sup>

### 3. Prediction Strategy
The **single-head** model predicts FA and MD through a shared output pathway. The **two-head** model uses separate output heads for FA and MD after shared feature extraction.

### 4. FA-Oriented SSIM Loss
The proposed loss adds an FA-specific SSIM term to the baseline loss:
```text
   L = L_base + λ_FA × (1 - SSIM_FA)
```
The experiments use:

- FA_SSIM = 0.0    # baseline
- FA_SSIM = 0.3    # FA-oriented SSIM
  

## Dataset
The experiments were conducted using data from the **Human Connectome Project
(HCP) Young Adult dataset**. The HCP minimally preprocessed dataset provides T1-weighted MRI together with diffusion-derived **FA and MD** maps, all at
an isotropic spatial resolution of **1.25 mm**.

The dataset was split at the subject level:

- 90 training subjects
- 30 validation subjects
- 40 test subjects
  
There is no subject overlap between the three splits. Paired T1-weighted MRI, FA, and MD patches were extracted for training and evaluation.

## Data Preprocessing
Update the target directory in hcp_preprocess.sh and run:
```text
./hcp_preprocess.sh
```
This script preprocesses all downloaded HCP data.

## Reproducing Experiments
Modify the corresponding options in config.py. For fair comparisons, change only the factor being evaluated and keep the other settings fixed.

### 1. Network Architecture
- MODEL = "CNN"
- MODEL = "UNet"
- MODEL = "TransUNet"

### 2. Patch Size
- BLOCK_SIZE = 64
- BLOCK_SIZE = 80
  
### 3. Prediction Head
- TWO_HEAD = True
- TWO_HEAD = False

### 4. FA-Oriented SSIM Loss
- FA_SSIM = 0.0
- FA_SSIM = 0.3

## How to Run

- **PreProcess**  
./hcp_preprocess.sh

- **Training**  
python train.py

- **Test**  
python test.py

## Notes
Please update the required **file paths and configuration options** in config.py before running the preprocessing, training, or testing scripts.