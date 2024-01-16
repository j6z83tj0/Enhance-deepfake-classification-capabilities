# Enhance-deepfake-classification-capabilities

Readme
This model employs three strategies to enhance the generalizability of the judgment: two color spaces, effective filters, and efficient threshold adjustments.

Setup
Operating System: Ubuntu 20.04
Environment: Anaconda (import requirements.txt)
Installation Steps:
Install PyTorch from pytorch.org.
Run the following command to install additional requirements:
bash
Copy code
pip install -r requirements.txt
Dataset
You can download the dataset using GenImage from here.

Dataset Structure:
lua
Copy code
Dataset
|-- GenImage
|   |-- sdv1.4
|   |   |-- train (48000 samples for training, 6000 for validation)
|   |   |   |-- 0_real
|   |   |   |   |-- xxxx.png
|   |   |   |   ...
|   |   |   |-- 1_fake
|   |   |   |   |-- yyyy.png
|   |   |   |   ...
|   |   |-- val (6000 samples for testing)
|   |   |   |-- 0_real
|   |   |   |   |-- xxxx.png
|   |   |   |   ...
|   |   |   |-- 1_fake
|   |   |   |   |-- yyyy.png
|   |   |   |   ...
|   |-- Midjourney
|   |-- ...
Training Your Model
The model is trained using a subset of the full training set (randomly selected). If you want to train on the entire dataset, modify the data/__init__.py file, specifically the get_dataset function.

Simple Command:
bash
Copy code
python train.py --name sdv14_fusingmodel_ycc_hsv_CONTOUR --dataroot ./Dataset/GenImage/stable_diffusion_v_1_4 
Additional Parameters:
--filter CONTOUR: Train images undergo filtering.
Model weights will be saved in the checkpoints directory.

Testing Your Model
Simple Command:
bash
Copy code
python eval.py --no_crop --batch_size 1 --eval_mode 
Additional Parameters:
--threshold 0.01: Adjust testing threshold.
--filter CONTOUR: Test images undergo filtering.
Test results will be saved in the results folder.
