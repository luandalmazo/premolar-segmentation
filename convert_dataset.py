from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import re 

TASK_NAME = "Tooth"
NNUNET_DATASET_ID = "900"
DATASET_NAME = f"Dataset{NNUNET_DATASET_ID}_{TASK_NAME}"
INPUT_DIR = "Images"
OUTPUT_DIR = join(nnUNet_raw, DATASET_NAME)
MAPPING_FILE = join(OUTPUT_DIR, "dataset_mapping.csv")

os.makedirs(join(OUTPUT_DIR, "imagesTr"), exist_ok=True)
os.makedirs(join(OUTPUT_DIR, "labelsTr"), exist_ok=True)

def parse_filename(filename):
    """
    Parses filenames like 'isabela_t0_24.nii.gz' or 'Pedro_14.nii.gz'.
    Returns (patient_id, time_point, tooth_id).
    """
    
    name = filename.split('.nii')[0]
    teeth_options = ['14', '15', '24', '25']
    tooth = next((t for t in teeth_options if f"_{t}" in name or name.endswith(f"_{t}")), None)
    
    time_match = re.search(r'_(t\d+)', name)
    time = time_match.group(1) if time_match else "t0"
    
    if time_match:
        patient_id = name.split(f"_{time}")[0]
    elif tooth:
        patient_id = name.split(f"_{tooth}")[0]
    else:
        patient_id = name.split('_')[0]
        
    return patient_id, time, tooth

def process_dataset():
    os.makedirs(join(OUTPUT_DIR, "imagesTr"), exist_ok=True)
    os.makedirs(join(OUTPUT_DIR, "labelsTr"), exist_ok=True)

    folders = [
        ("Author1_Seg", "Author1_Tomografias", "Author1"),
        ("Author2-Seg", "Author2-Tomografias", "Author2"),
        ("Author3-Seg", "Author3-Tomografias", "Author3")
    ]
    
    tooth_pairs = {
        "15": "14", 
        "25": "24"  
    }
    
    case_id_counter = 1
    mapping_data = []
    train_case_ids = []
    
    stats = {
        "aluna": {}, 
        "patient_teeth": {}, 
        "patient_times": {}  
    }

    for seg_dir, tomo_dir, aluna_name in folders:
        seg_path = join(INPUT_DIR, seg_dir)
        tomo_path = join(INPUT_DIR, tomo_dir)

        if not isdir(seg_path): continue
        if aluna_name not in stats["aluna"]:
            stats["aluna"][aluna_name] = {"patients": set(), "teeth": {"14": 0, "15": 0, "24": 0, "25": 0}, "total_scans": 0}
            
        seg_files = [f for f in os.listdir(seg_path) if f.endswith(".nii.gz")]
        
        for seg_f in seg_files:
            patient, time, tooth = parse_filename(seg_f)
            
            # 1:2
            tomo_f = seg_f 
            if not isfile(join(tomo_path, tomo_f)):
                if aluna_name == "Author3":
                    target_tooth = tooth_pairs[tooth]
                
                    expected_tomo_name = seg_f.replace(f"_{tooth}", f"_{target_tooth}")
                    
                    if isfile(join(tomo_path, expected_tomo_name)):
                        tomo_f = expected_tomo_name
                    else:
                        prefix = f"{patient}_{time}" if "_t" in seg_f else patient
                        potential = [f for f in os.listdir(tomo_path) 
                                    if f.startswith(prefix) and f"_{target_tooth}" in f]
                        
                        if potential:
                            tomo_f = potential[0]
                        else:
                            print(f"Error finding tomo for {seg_f} in {tomo_path}")
                            continue
                else:
                    continue
                
            stats["aluna"][aluna_name]["patients"].add(patient)
            stats["aluna"][aluna_name]["total_scans"] += 1
            if tooth in stats["aluna"][aluna_name]["teeth"]:
                stats["aluna"][aluna_name]["teeth"][tooth] += 1
            
            if patient not in stats["patient_teeth"]:
                stats["patient_teeth"][patient], stats["patient_times"][patient] = set(), set()
            
            if tooth: stats["patient_teeth"][patient].add(tooth)
            stats["patient_times"][patient].add(time)

            case_name = f"Tooth_{case_id_counter:04d}"
            tomo_sitk = sitk.ReadImage(join(tomo_path, tomo_f))
            seg_sitk = sitk.ReadImage(join(seg_path, seg_f))
            
            # Binarization
            seg_sitk = sitk.BinaryThreshold(seg_sitk, lowerThreshold=1, upperThreshold=1000, insideValue=1, outsideValue=0)
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(tomo_sitk)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetTransform(sitk.Transform())
            
            seg_fixed = resampler.Execute(seg_sitk)
            
            sitk.WriteImage(tomo_sitk, join(OUTPUT_DIR, "imagesTr", f"{case_name}_0000.nii.gz"))
            sitk.WriteImage(seg_fixed, join(OUTPUT_DIR, "labelsTr", f"{case_name}.nii.gz"))
            
            mapping_data.append(f"{aluna_name},{seg_f},{case_name},{patient}")
            train_case_ids.append(case_name)
            case_id_counter += 1

    with open(MAPPING_FILE, "w") as f:
        f.write("Author,Original_File,CaseID,Patient\n")
        for line in mapping_data:
            f.write(line + "\n")

    print("\n" + "="*40)
    print("Statistics")
    print("="*40)
    for aluna, data in stats["aluna"].items():
        print(f"\nAuthor: {aluna}")
        print(f"  - Patients (Unique): {len(data['patients'])}")
        print(f"  - Total Scan/Seg Pairs: {data['total_scans']}")
        print(f"  - Teeth: {data['teeth']}")
        
    inc_teeth = [p for p, t in stats["patient_teeth"].items() if len(t) < 4]
    inc_times = [p for p, t in stats["patient_times"].items() if len(t) < 2]
    
    print("\n" + "-"*40)
    print(" Summary")
    print("-"*40)
    print(f"Total nnU-Net Cases:      {len(train_case_ids)}")
    print(f"Patients < 4 teeth:       {len(inc_teeth)}")
    print(f"Patients < 2 time points: {len(inc_times)}")
    print("="*40 + "\n")

    return train_case_ids
        
def convert_to_nnunet_format(input_images, input_masks, output_folder):
    pass 

if __name__ == "__main__":
    print("Starting dataset conversion...")
    train_cases = process_dataset()

    generate_dataset_json(
        output_folder=OUTPUT_DIR,
        channel_names={0: "CBCT"}, 
        labels={"background": 0, "tooth": 1},
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=TASK_NAME
    )
    print(f"Dataset saved: {OUTPUT_DIR}")