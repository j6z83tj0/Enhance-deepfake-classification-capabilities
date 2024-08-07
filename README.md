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
```
Dataset  
|-- GenImage  
|   |-- sdv1.4  
|   |   |-- train   
|   |   |   |-- 0_real(48000 samples for training, 6000 for validation)  
|   |   |   |   |-- xxxx.png  
|   |   |   |   |...  
|   |   |   |-- 1_fake(48000 samples for training, 6000 for validation)  
|   |   |   |   |-- yyyy.png  
|   |   |   |   |...  
|   |   |-- val  
|   |   |   |-- 0_real (6000 samples for testing)  
|   |   |   |   |-- xxxx.png  
|   |   |   |   |...  
|   |   |   |-- 1_fake (6000 samples for testing)  
|   |   |   |   |-- yyyy.png  
|   |   |   |   |...  
|   |-- Midjourney  
|   |-- |...  
```
## Quick start
### Run on a single image
This command runs the model on a single image, and outputs the uncalibrated prediction.
```bash
python demo.py -f examples/real.png -m checkpoints/sdv14_fusingmodel_ycc_hsv_EDGE_ENHANCE/model_epoch_latest.pth --filter EDGE_ENHANCE
```

## Train Your Model

This model is trained using a subset of the full training set (randomly selected). If you want to train on the entire dataset, please modify the `get_dataset` function in the `data/__init__.py` file.

### Simple Command:
```bash
python train.py --name sdv14_fusingmodel_ycc_hsv_CONTOUR --dataroot /Dataset/GenImage/stable_diffusion_v_1_4
```
Additional Parameters:  
--filter CONTOUR: Training images undergo filtering  
(None,"CONTOUR","DETAIL","EDGE_ENHANCE","EMBOSS","FIND_EDGES","SMOOTH","SHARPEN","UnsharpMask","ModeFilter","GAUSSIAN_BLUR").  
Model weights will be saved in the checkpoints directory.
## Test Your Model
### Simple Command:
```bash
python eval.py --no_crop --batch_size 1 --eval_mode
```
Additional Parameters:  
--threshold 0.01: Adjust the testing threshold.  
--filter CONTOUR: Test images undergo filtering.  
Test results will be saved in the results folder.


