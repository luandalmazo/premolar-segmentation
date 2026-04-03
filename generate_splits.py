import pandas as pd
import json
import random
from pathlib import Path

DATASET_NAME = "Dataset903_Tooth"
BASE_PATH = Path(f"/scratch/nnUNet_raw/{DATASET_NAME}") 
PREPROCESSED_PATH = Path(f"/scratch/nnUNet_preprocessed/{DATASET_NAME}")
CSV_PATH = BASE_PATH / "dataset_mapping_903.csv"
OUTPUT_SPLIT = PREPROCESSED_PATH / "splits_final.json"

def generate_splits():

    # Solving mapping issue: Removing '.nii.gz' from CaseID
    df = pd.read_csv(CSV_PATH)
    df['CaseID'] = df['CaseID'].str.replace('.nii.gz', '', regex=False)
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] CSV mapping cleaned and updated at {CSV_PATH}")

    # Unique ID: Combining Author + Patient 
    df['Unique_ID'] = df['Author'].astype(str) + "_" + df['Patient'].astype(str)
    unique_identities = sorted(df['Unique_ID'].unique().tolist())
    random.seed(42) 
    random.shuffle(unique_identities)

    total_patients = len(unique_identities)
    print(f"--- DATASET SUMMARY ---")
    print(f"Total Unique Patients Found: {total_patients}")
    print(f"Total Volumes in CSV: {len(df)}")
    print(f"-----------------------\n")

    blocks = []
    block_size = 5
    
    for i in range(0, total_patients, block_size):
        blocks.append(unique_identities[i:i + block_size])

    splits = []

    print(f"--- FOLD DISTRIBUTION DETAILS ---")

    for i in range(5):
        test_block_idx = i
        val_block_idx = (i + 1) % 5
        train_block_indices = [j for j in range(5) if j != test_block_idx and j != val_block_idx]
        
        test_ids = blocks[test_block_idx]
        val_ids = blocks[val_block_idx]
        train_ids = []
        for idx in train_block_indices:
            train_ids.extend(blocks[idx])

        # Original images + data augmentation            
        train_cases = df[df['Unique_ID'].isin(train_ids)]['CaseID'].tolist()

        # Only original images
        val_cases = df[(df['Unique_ID'].isin(val_ids)) & (df['Derived_From'] == 'Original')]['CaseID'].tolist()
        test_cases = df[(df['Unique_ID'].isin(test_ids)) & (df['Derived_From'] == 'Original')]['CaseID'].tolist()
        
        splits.append({
            "train": sorted(train_cases),
            "val": sorted(val_cases),
            "test": sorted(test_cases)
        })

        print(f"FOLD {i}:")
        print(f"  [TRAIN] Patients: {len(train_ids):<2} | Total Volumes (Inc. Aug): {len(train_cases)}")
        print(f"  [VAL]   Patients: {len(val_ids):<2} | Total Volumes (Original): {len(val_cases)}")
        print(f"  [TEST]  Patients: {len(test_ids):<2} | Total Volumes (Original): {len(test_cases)}")
        print("-" * 30)

    OUTPUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_SPLIT, 'w') as f:
        json.dump(splits, f, indent=4)

    print(f"\nSaved to {OUTPUT_SPLIT}")

if __name__ == "__main__":
    generate_splits()