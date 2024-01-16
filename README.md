# Enhance-deepfake-classification-capabilities

# Project Overview

This model enhances judgment generality through three strategies: leveraging two color spaces, implementing effective filters, and employing efficient threshold adjustments.

## Setup
- **Operating System:** Ubuntu 20.04
- **Environment:** Anaconda (import `requirements.txt`)

### Installation
1. Install PyTorch from [pytorch.org](https://pytorch.org).
2. Run `pip install -r requirements.txt` for additional dependencies.

## Dataset
Download the dataset from [GenImage](https://github.com/GenImage-Dataset/GenImage).

### Dataset Structure:
Dataset
|-- GenImage
| |-- sdv1.4
| | |-- train (48000 samples for training, 6000 for validation)
| | | ...
| | |-- val (6000 samples for testing)
| | | ...
| |-- Midjourney
| |-- ...


## Training Your Model
- Subset training is used; modify `data/__init__.py` for the entire dataset.
- Train with:
  ```bash
  python train.py --name sdv14_fusingmodel_ycc_hsv_CONTOUR --dataroot ./Dataset/GenImage/stable_diffusion_v_1_4 
Testing Your Model
Test with:
bash
Copy code
python eval.py --no_crop --batch_size 1 --eval_mode 
Additional Parameters:
--threshold 0.01: Adjust the testing threshold.
--filter CONTOUR: Test images undergo filtering.
Test results will be saved in the results folder.
