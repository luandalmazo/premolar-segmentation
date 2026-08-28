# Segmentation of Premolar Teeth Using Image Pre-processing Techniques and Deep Learning

This project aims at the automated segmentation of upper premolar teeth (14, 15, 24, and 25) using Cone Beam Computed Tomography (CBCT) images.

---

## 💻 For Developers

This guide describes the steps required to replicate the processing, training, and evaluation performed.

**Install the dependencies**

```bash
pip install -r requirements.txt
```

**Install nnU-Net**

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

### Pretrained Checkpoints

Pretrained model checkpoints are available for download below. Each folder contains the checkpoints of the fold with the best results.

- [nnU-Net checkpoints](https://drive.google.com/drive/folders/1p40SNDsTysR7Bkl3_OfqckU2fip1h1Sc?usp=sharing)
- [nnU-Net ResNet checkpoints](https://drive.google.com/drive/folders/10wGI0_L4LN5kStCGFCX55olkQfJUY64o?usp=sharing)

### 1. Data Preparation and Augmentation

Before starting training, the dataset must be converted to the nnU-Net standard format.

#### Dataset Conversion

Make sure the raw data is organized and run the conversion to the `Dataset903_Tooth` standard:

    python convert_dataset.py

#### Manual Data Augmentation

Run the script to generate an 11x augmentation of the volumes:

    python manual_augmentation.py

#### Splits Generation

To ensure reproducibility of the cross-validation, generate the fold split files:

    python generate_splits.py

### 2. Planning and Pre-processing

Planning extracts the dataset properties to configure the networks.

#### nnU-Net

    # Standard nnU-Net v2 planning for 80GB of VRAM
    nnUNetv2_plan_and_preprocess -d 903 -c 3d_fullres

#### nnU-Net ResNet (if the images have already been preprocessed for training the native nnU-Net)

    nnUNetv2_plan_experiment -d 903 \
        -pl nnUNetPlannerResEncL \
        -gpu_memory_target 80 \
        -overwrite_plans_name nnUNetResEncUNetPlans_L_80G

### 3. Architecture Training

Since the dataset has already undergone a manual data augmentation process, we use the `nnUNetTrainerNoDA` trainer to disable the redundant native transformations.

#### Native nnU-Net

    # Training of Fold 0 (repeat for folds 0 to 4)
    nnUNetv2_train 903 3d_fullres 0 -tr nnUNetTrainerNoDA

#### nnU-Net ResNet

For the ResNet variant, make sure the specific plan for the residual encoder has been generated:

    # Training of Fold 0 using the ResNet architecture
    nnUNetv2_train 903 3d_fullres 0 -p nnUNetResEncUNetPlans_L_80G -tr nnUNetTrainerNoDA

### 4. Prediction and Evaluation

After training is complete, follow these steps to validate the models on the test data.

#### Test Organization

Prepare the output directories:

    python organize_test_folders.py

#### Prediction

Run inference using the best checkpoint:

    # Native nnU-Net
    
    nnUNetv2_predict -i [INPUT_DIRECTORY] -o [OUTPUT_DIRECTORY] -d 903 -c 3d_fullres -f [FOLDS] -tr [TRAINER_USED]
    
    # nnU-Net ResNet
    
    nnUNetv2_predict -i [INPUT_DIRECTORY] -o [OUTPUT_DIRECTORY] -d 903 -c 3d_fullres -f [FOLDS] -tr [TRAINER_USED]-p nnUNetResEncUNetPlans_L_80G
    

#### Metrics Generation

Extract the quantitative results (Dice, IoU, HD95, ASSD):

    python3 generate_test_results.py
    python3 analyze_results.py

### 🎥 3D Slicer Demonstration

Watch our demonstration video of the model integrated into 3D Slicer:

👉 Watch the video [HERE](https://drive.google.com/file/d/1wNqsYPZWvQr9I3iy7mbz6cN8zq6dLlKF/view?usp=sharing)
