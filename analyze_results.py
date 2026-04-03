import pandas as pd

CSV_PATH = "master_metrics_medpy_903.csv"

def run_analysis():
    df = pd.read_csv(CSV_PATH)
    
    print("="*60)
    print("           TCC FINAL STATISTICAL SUMMARY")
    print("="*60)

    # Global Dice (considering all classes)
    overall_dice = df['Dice_Global'].mean()
    overall_std = df['Dice_Global'].std()
    print(f"\nOVERALL MEAN DICE: {overall_dice:.4f} ± {overall_std:.4f}")
    
    overall_iou = df['IoU_Global'].mean()
    overall_iou_std = df['IoU_Global'].std()
    
    print(f"OVERALL MEAN IoU: {overall_iou:.4f} ± {overall_iou_std:.4f}")

    print("\n PER TOOTH:")
    tooth_cols = ['Dice_14', 'Dice_15', 'Dice_24', 'Dice_25', 'IoU_14',	'IoU_15', 'IoU_24',	'IoU_25', 'HD95_14', 'HD95_15',	'HD95_24' , 'HD95_25', 'ASD_14', 'ASD_15', 'ASD_24','ASD_25']
    tooth_means = df[tooth_cols].mean()
    for tooth, val in tooth_means.items():
        std = df[tooth].std()
        print(f"  {tooth}: {val:.4f} ± {std:.4f}")

    print("\nMEAN DICE PER SIDE:")
    left_df = df[df['Side'] == 'Left (14,15)'][['Dice_14', 'Dice_15']].mean(axis=1)
    right_df = df[df['Side'] == 'Right (24,25)'][['Dice_24', 'Dice_25']].mean(axis=1)

    print(f"  Left Side (14, 15):  {left_df.mean():.4f} ± {left_df.std():.4f}")
    print(f"  Right Side (24, 25): {right_df.mean():.4f} ± {right_df.std():.4f}")
    
    print ("\nMEAN IOU PER SIDE:")
    left_iou = df[df['Side'] == 'Left (14,15)'][['IoU_14', 'IoU_15']].mean(axis=1)
    right_iou = df[df['Side'] == 'Right (24,25)'][['IoU_24', 'IoU_25']].mean(axis=1)
    
    print(f"  Left Side (14, 15):  {left_iou.mean():.4f} ± {left_iou.std():.4f}")
    print(f"  Right Side (24, 25): {right_iou.mean():.4f} ± {right_iou.std():.4f}")
    
    print("\nMEAN HD95 PER SIDE:")
    left_hd95 = df[df['Side'] == 'Left (14,15)'][['HD95_14', 'HD95_15']].mean(axis=1)
    right_hd95 = df[df['Side'] == 'Right (24,25)'][['HD95_24', 'HD95_25']].mean(axis=1)
    print(f"  Left Side (14, 15):  {left_hd95.mean():.4f} ± {left_hd95.std():.4f}")
    print(f"  Right Side (24, 25): {right_hd95.mean():.4f} ± {right_hd95.std():.4f}")
    
    print("\nMEAN ASD PER SIDE:")
    left_asd = df[df['Side'] == 'Left (14,15)'][['ASD_14', 'ASD_15']].mean(axis=1)
    right_asd = df[df['Side'] == 'Right (24,25)'][['ASD_24', 'ASD_25']].mean(axis=1)
    print(f"  Left Side (14, 15):  {left_asd.mean():.4f} ± {left_asd.std():.4f}")
    print(f"  Right Side (24, 25): {right_asd.mean():.4f} ± {right_asd.std():.4f}")

    print("\nMEAN DICE, IOU, HD95 AND ASD PER FOLD (CROSS-VALIDATION):")
    fold_stats = df.groupby('Fold')[['Dice_Global', 'IoU_Global', 'ASD_mm Global', 'HD95_mm_Global']].agg(['mean', 'std'])
    print(fold_stats.to_string())
    
    print("\nMEAN DISTANCE METRICS ")
    distance_cols = ['ASD_mm Global', 'HD95_mm_Global']
    distance_means = df[distance_cols].mean()
    for metric, val in distance_means.items():
        std = df[metric].std()
        print(f"  {metric}: {val:.4f} ± {std:.4f}")
    

    print("\n" + "="*60)

if __name__ == "__main__":
    run_analysis()