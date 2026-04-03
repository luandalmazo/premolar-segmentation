import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
)

BASE_PATH = Path("/scratch/nnUNet_raw")
DIR_902 = BASE_PATH / "Dataset902_Tooth"
DIR_903 = BASE_PATH / "Dataset903_Tooth"
(DIR_903 / "imagesTr").mkdir(parents=True, exist_ok=True)
(DIR_903 / "labelsTr").mkdir(parents=True, exist_ok=True)

df_old = pd.read_csv(DIR_902 / "dataset_mapping.csv")

def apply_augmentation(img_data, lbl_data):
    data_dict = {
        "image": img_data[None], 
        "label": lbl_data[None]
    }

    aug = Compose([
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"), 
            prob=1.0,                     
            rotate_range=(15 * (3.1415/180), 15 * (3.1415/180), 15 * (3.1415/180)),
            padding_mode="zeros"
        ),
        RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
    ])

    output = aug(data_dict)
    
    return output["image"][0], output["label"][0]

if __name__ == "__main__":
    new_mapping_rows = []

    for _, row in df_old.iterrows():
        case_name = row['CaseID'] + ".nii.gz" 
        case_name_img = row['CaseID'] + "_0000.nii.gz" 
        
        img_path = DIR_902 / "imagesTr" / case_name_img
        lbl_path = DIR_902 / "labelsTr" / case_name
        
        img_nib = nib.load(img_path)
        lbl_nib = nib.load(lbl_path)
        
        # Copying original to 903
        nib.save(img_nib, DIR_903 / "imagesTr" / case_name_img)
        nib.save(lbl_nib, DIR_903 / "labelsTr" / case_name)
        
        # Adding new mapping
        new_row = row.to_dict()
        new_row['Derived_From'] = 'Original'
        new_mapping_rows.append(new_row)
        
        # Augmentation
        img_data = img_nib.get_fdata()
        lbl_data = lbl_nib.get_fdata()
        
        for i in range(1, 11):
            aug_img, aug_lbl = apply_augmentation(img_data, lbl_data)
            aug_file_name = row['CaseID'] + f"aug{i}" + "_0000.nii.gz"
            aug_segmentation_name =  row['CaseID'] + f"aug{i}" + ".nii.gz"
            
            nib.save(nib.Nifti1Image(aug_img, img_nib.affine, img_nib.header), 
                    DIR_903 / "imagesTr" / aug_file_name)
            nib.save(nib.Nifti1Image(aug_lbl.astype(np.uint8), lbl_nib.affine, lbl_nib.header), 
                    DIR_903 / "labelsTr" / aug_segmentation_name)
            
            aug_mapping_row = row.to_dict()
            aug_mapping_row['CaseID'] = aug_segmentation_name
            aug_mapping_row['Derived_From'] = case_name
            new_mapping_rows.append(aug_mapping_row)

    df_new = pd.DataFrame(new_mapping_rows)
    df_new.to_csv(DIR_903 / "dataset_mapping_903.csv", index=False)
    print(f"Done")