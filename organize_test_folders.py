import json
import shutil
from pathlib import Path

DATASET_NAME = "Dataset903_Tooth"
RAW_DIR = Path(f"/scratch/nnUNet_raw/{DATASET_NAME}/imagesTr")
SPLITS_FILE = Path(f"/scratch/nnUNet_preprocessed/{DATASET_NAME}/splits_final.json")
BASE_TEST_DIR = Path(f"/scratch/nnUNet_raw/{DATASET_NAME}/testFolds")

def organize():
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)

    for i, fold in enumerate(splits):
        fold_dir = BASE_TEST_DIR / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Organizing Fold {i}...")
        test_cases = fold['test']
        
        for case in test_cases:
            filename = f"{case}_0000.nii.gz"
            src = RAW_DIR / filename
            dst = fold_dir / filename
            
            if src.exists():
                shutil.copy(src, dst)
            else:
                print(f"  {filename} not found in imagesTr")

    print(f"\nDone, created at {BASE_TEST_DIR}")

if __name__ == "__main__":
    organize()