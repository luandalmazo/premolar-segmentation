import pandas as pd
import nibabel as nib
import numpy as np
from pathlib import Path
from medpy.metric import binary as medpy_metrics

DATASET_NAME = "Dataset903_Tooth"
RESULTS_DIR = Path(f"/scratch/nnUNet_results/{DATASET_NAME}/inference_results")
LABELS_DIR = Path(f"/scratch/nnUNet_raw/{DATASET_NAME}/labelsTr")
MAPPING_CSV = Path(f"/scratch/nnUNet_raw/{DATASET_NAME}/dataset_mapping_903.csv")

def get_medpy_metrics(pred, gt, voxel_spacing):
    """Calculate Dice, IoU, HD95, and ASD using MedPy for binary masks."""

    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 0.0, 0.0, 0.0, 0.0 
    
    dice = medpy_metrics.dc(pred, gt)
    iou = medpy_metrics.jc(pred, gt)
    hd95 = medpy_metrics.hd95(pred, gt, voxelspacing=voxel_spacing)
    asd = medpy_metrics.asd(pred, gt, voxelspacing=voxel_spacing)
    
    return dice, iou, hd95, asd

def run_evaluation():
    df = pd.read_csv(MAPPING_CSV)
    test_df = df[df['Derived_From'] == 'Original'].copy()
    all_results = []

    for fold in range(5):
        fold_dir = RESULTS_DIR / f"fold_{fold}"
        if not fold_dir.exists(): continue
        
        print(f"Processing Fold {fold}...")
    
        for pred_file in fold_dir.glob("*.nii.gz"):
            case_id = pred_file.name.replace(".nii.gz", "")
            label_file = LABELS_DIR / f"{case_id}.nii.gz"
            
            meta = test_df[test_df['CaseID'] == case_id].iloc[0]
            original_filename = str(meta['Original_Files']).lower()

            if any(x in original_filename for x in ["_14", "_15"]):
                side = "Left (14,15)"
            elif any(x in original_filename for x in ["_24", "_25"]):
                side = "Right (24,25)"
    
            group = "Follow-up (T0/T1)" if any(x in original_filename for x in ["t0", "t1"]) else "Single-Session"
            
            pred_obj = nib.load(pred_file)
            gt_obj = nib.load(label_file)            
            spacing = pred_obj.header.get_zooms()
            pred_data = pred_obj.get_fdata().astype(np.uint8)
            gt_data = gt_obj.get_fdata().astype(np.uint8)
            
            labels_present = np.unique(gt_data)
            # We don't consider the background (label 0)
            labels_present = labels_present[labels_present > 0]  

            dices_list, ious_list, hd95_list, asds_list = [], [], [], []

            label_metrics = {}
            for label in labels_present:
                d, i, h, a = get_medpy_metrics(pred_data == label, gt_data == label, spacing)
                label_metrics[int(label)] = d, i, h, a
                dices_list.append(d)
                ious_list.append(i)
                hd95_list.append(h)
                asds_list.append(a)

            dice_global_real = np.mean(dices_list) if dices_list else 0.0
            iou_global_real = np.mean(ious_list) if ious_list else 0.0
            hd95_global_real = np.mean(hd95_list) if hd95_list else 0.0
            asd_global_real = np.mean(asds_list) if asds_list else 0.0

            # If the file contains labels 14 and 15, we assign their Dice scores to dice_14 and dice_15 respectively.
            # But we don't want to assign any value to dice_24 and dice_25, so we set them to None. The opposite applies if the file contains labels 24 and 25.
            tooth_14_metrics = label_metrics.get(1, (None, None, None, None))
            tooth_15_metrics = label_metrics.get(2, (None, None, None, None))
            tooth_24_metrics = label_metrics.get(3, (None, None, None, None))
            tooth_25_metrics = label_metrics.get(4, (None, None, None, None))
                
            # Grouped Metrics (Left vs Right)
            dice_L, iou_L, hd_95_L, asd_L = get_medpy_metrics(np.isin(pred_data, [1, 2]), np.isin(gt_data, [1, 2]), spacing)
            dice_R, iou_R, hd_95_R, asd_R = get_medpy_metrics(np.isin(pred_data, [3, 4]), np.isin(gt_data, [3, 4]), spacing)
        
            all_results.append({
                "CaseID": case_id,
                "Patient": meta['Patient'],
                "Group": group, 
                "Dice_Global": dice_global_real,
                "IoU_Global": iou_global_real,
                "ASD_mm": asd_global_real,
                "Dice_14": tooth_14_metrics[0],
                "Dice_15": tooth_15_metrics[0],
                "Dice_24": tooth_24_metrics[0],
                "Dice_25": tooth_25_metrics[0],
                "IoU_14": tooth_14_metrics[1],
                "IoU_15": tooth_15_metrics[1],
                "IoU_24": tooth_24_metrics[1],
                "IoU_25": tooth_25_metrics[1],
                "HD95_14": tooth_14_metrics[2],
                "HD95_15": tooth_15_metrics[2],
                "HD95_24": tooth_24_metrics[2],
                "HD95_25": tooth_25_metrics[2],
                "ASD_14": tooth_14_metrics[3],
                "ASD_15": tooth_15_metrics[3],
                "ASD_24": tooth_24_metrics[3],
                "ASD_25": tooth_25_metrics[3],
                "Dice_Left(14,15)": dice_L,
                "IoU_Left(14,15)": iou_L,
                "HD95_Left(14,15)": hd_95_L,
                "ASD_Left(14,15)": asd_L,
                "Dice_Right(24,25)": dice_R,
                "IoU_Right(24,25)": iou_R,
                "HD95_Right(24,25)": hd_95_R,
                "ASD_Right(24,25)": asd_R,
                "HD95_mm": hd95_global_real,
                "Fold": fold,
                "Side": side,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("master_metrics_medpy_903.csv", index=False)
    print("\nDone.")

if __name__ == "__main__":
    run_evaluation()