#!/usr/bin/env python3

import os
import argparse
import sys
import json

# Import the to_nnunet_format function from to_nnunet.py
from to_nnunet import to_nnunet_format

def process_kvasirseg():
    """
    Process the Kvasir-SEG dataset by unzipping the dataset.

    Returns:
    - data_paths (dict): Dictionary containing paths to images and masks.
    """
    print("\nProcessing Kvasir-SEG dataset...")

    # Prompt for zip file path
    zip_path = input("Enter the path to the Kvasir-SEG zip file (e.g., '/data/open_dataset/kvasir-seg.zip'): ").strip()
    while not os.path.isfile(zip_path):
        print(f"Error: ZIP file '{zip_path}' does not exist.")
        zip_path = input("Please enter a valid path to the Kvasir-SEG zip file: ").strip()

    # Prompt for extraction directory
    extract_to_dir = input("Enter the directory to extract the zip file to (default is zip_path without '.zip'): ").strip()
    if not extract_to_dir:
        extract_to_dir = zip_path.rstrip('.zip')

    if os.path.exists(extract_to_dir):
        print(f"Extraction directory '{extract_to_dir}' already exists. Skipping unzip.")
    else:
        os.makedirs(extract_to_dir, exist_ok=True)
        print(f"Extracting ZIP file '{zip_path}' to '{extract_to_dir}'...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        print(f"ZIP file extracted to: {extract_to_dir}")

    # Define images and masks directories
    images_dir = os.path.join(extract_to_dir, "Kvasir-SEG", "images")
    masks_dir = os.path.join(extract_to_dir, "Kvasir-SEG", "masks")

    data_paths = {
        "images_dir": images_dir,
        "masks_dir": masks_dir
    }
    return data_paths

def process_cvc():
    """
    Process the CVC dataset by constructing the images and masks directories based on the base dataset path.

    Returns:
    - data_paths (dict): Dictionary containing paths to images and masks.
    """
    print("\nProcessing CVC dataset...")

    # Prompt for base dataset directory
    base_dir = input("Enter the base path to the CVC dataset (e.g., '/data/open_dataset/cvc'): ").strip()
    while not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        base_dir = input("Please enter a valid base path to the CVC dataset: ").strip()

    # Construct images and masks directories
    images_dir = os.path.join(base_dir, '1', 'PNG', 'Original')
    masks_dir = os.path.join(base_dir, '1', 'PNG', 'Ground Truth')

    # Check if constructed directories exist
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(masks_dir):
        print(f"Error: Masks directory '{masks_dir}' does not exist.")
        sys.exit(1)

    data_paths = {
        "images_dir": images_dir,
        "masks_dir": masks_dir
    }
    return data_paths



def process_bkai():
    """
    Process the BKAI dataset by constructing the images and masks directories based on the base dataset path.

    Returns:
    - data_paths (dict): Dictionary containing paths to images and masks.
    """
    
    print("\nProcessing BKAI dataset...")

    base_dir = input("Enter the base path to the BKAI dataset (e.g., '/data/open_dataset/bkai-igh-neopolyp'): ").strip()
    while not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        base_dir = input("Please enter a valid base path to the BKAI dataset: ").strip()

    images_dir = os.path.join(base_dir, 'train', 'train')
    masks_dir = os.path.join(base_dir, 'train_gt', 'train_gt')

    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(masks_dir):
        print(f"Error: Masks directory '{masks_dir}' does not exist.")
        sys.exit(1)

    data_paths = {
        "images_dir": images_dir,
        "masks_dir": masks_dir
    }
    return data_paths


def process_busi():
    """
    Process the BUSI dataset by mapping image and mask filenames.

    Returns:
    - data_paths (dict): Dictionary containing base directory and class directories.
    """
    print("\nProcessing BUSI dataset...")

    # Prompt for base dataset directory
    base_dir = input("Enter the base path to the BUSI dataset (e.g., '/data/open_dataset/busi/1/Dataset_BUSI_with_GT'): ").strip()
    while not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        base_dir = input("Please enter a valid base path to the BUSI dataset: ").strip()

    # The images and masks are under 'benign', 'malignant', and possibly 'normal' directories
    class_dirs = {}
    for class_name in ['benign', 'malignant', 'normal']:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            class_dirs[class_name] = class_dir

    if not class_dirs:
        print("Error: No valid class directories ('benign', 'malignant', 'normal') found in the base directory.")
        sys.exit(1)

    data_paths = {
        "base_dir": base_dir,
        "class_dirs": class_dirs
    }
    return data_paths


def dataset_json_matching(dataset_name, medvlsm_path, data_path, dataset_number):
    """
    Matches train and test JSON files with images and masks, and creates a new dataset JSON.

    Parameters:
    - dataset_name (str): Name of the dataset (e.g., 'kvasirseg', 'cvc', 'bkai', 'busi')
    - medvlsm_path (str): Path to the medvlsm directory.
    - data_path (dict): Data paths containing base directory and class directories.
    - dataset_number (int): Dataset number for nnUNet format.
    """
    json_files = ["train.json", "test.json"]

    dataset_json = {
        "channel_names": {},
        "labels": {},
        "numTraining": 0,
        "file_ending": ".nii.gz",
        "name": "",
        "imagesTr": [],
        "labelsTr": [],
        "imagesTs": [],
        "labelsTs": []
    }

    # Set dataset-specific parameters
    if dataset_name.lower() == "kvasirseg":
        json_dir = os.path.join(medvlsm_path, "data", "kvasir_polyp", "anns")
        dataset_json["channel_names"] = {
            "0": "gastrointestinal_polyp"
        }
        dataset_json["labels"] = {
            "background": 0,
            "polyp": 1
        }
        dataset_json["name"] = f"Dataset{str(dataset_number).zfill(3)}_{dataset_name}"

    elif dataset_name.lower() == "cvc":
        json_dir = os.path.join(medvlsm_path, "data", "cvc300_polyp", "anns")
        dataset_json["channel_names"] = {
            "0": "gastrointestinal_polyp"
        }
        dataset_json["labels"] = {
            "background": 0,
            "polyp": 1
        }
        dataset_json["name"] = f"Dataset{str(dataset_number).zfill(3)}_{dataset_name}"
    elif dataset_name.lower() == "bkai":
        json_dir = os.path.join(medvlsm_path, "data", "bkai_polyp", "anns")
        dataset_json["channel_names"] = {
            "0": "colonoscopy_polyp"
        }
        dataset_json["labels"] = {
            "background": 0,
            "neoplastic_polyp": 1,
            "non-neoplastic_polyp": 2
        }
        dataset_json["name"] = f"Dataset{str(dataset_number).zfill(3)}_{dataset_name}"
    elif dataset_name.lower() == "busi":
        json_dir = os.path.join(medvlsm_path, "data", "busi", "anns")
        dataset_json["channel_names"] = {
            "0": "breast_ultrasound"
        }
        dataset_json["labels"] = {
            "background": 0,
            "tumor": 1
        }
        dataset_json["name"] = f"Dataset{str(dataset_number).zfill(3)}_{dataset_name}"
    else:
        print(f"Dataset '{dataset_name}' not recognized.")
        sys.exit(1)

    patient_id = 1  # Starting patient ID
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"\n{json_file} does not exist in {json_dir}.")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            img_name = entry["img_name"]  # e.g., "100_benign.png"
            mask_name = entry["mask_name"]
            prompts = entry["prompts"]

            if dataset_name.lower() == "busi":
                # For BUSI dataset, map img_name to actual filename
                base_name = os.path.splitext(img_name)[0]  # "100_benign"
                parts = base_name.split('_')  # ["100", "benign"]
                if len(parts) != 2:
                    print(f"Unexpected img_name format: {img_name}")
                    continue
                number, class_name = parts
                class_name = class_name.lower()
                ori_img_name = f"{class_name} ({number}).png"
                ori_mask_name = f"{class_name} ({number})_mask.png"
                class_dir = data_path["class_dirs"].get(class_name)
                if not class_dir:
                    print(f"Class directory for '{class_name}' not found.")
                    continue
                # Store the class directory in entry for later use
                entry["class_dir"] = class_dir
            else:
                ori_img_name = img_name
                ori_mask_name = mask_name

            # Create new image name
            new_image_name = f"{dataset_name}{str(patient_id).zfill(3)}_0000"

            # Build dictionary entry
            sample_entry = {
                "image_name": new_image_name,
                "ori_img_name": ori_img_name,
                "ori_mask_name": ori_mask_name,
                "prompts": prompts
            }

            if dataset_name.lower() == "busi":
                sample_entry["class_dir"] = class_dir

            if json_file == "train.json":
                dataset_json["imagesTr"].append(sample_entry)
                dataset_json["labelsTr"].append(sample_entry)
            elif json_file == "test.json":
                dataset_json["imagesTs"].append(sample_entry)
                dataset_json["labelsTs"].append(sample_entry)

            patient_id += 1

    # Set numTraining
    dataset_json["numTraining"] = len(dataset_json["imagesTr"])

    # Save the dataset JSON
    dataset_id = str(dataset_number).zfill(3)
    nnunet_dataset_dir = os.path.join(
        "..",
        f"data/nnUNet_raw/Dataset{dataset_id}_{dataset_name}"
    )
    os.makedirs(nnunet_dataset_dir, exist_ok=True)

    dataset_json_path = os.path.join(nnunet_dataset_dir, "dataset.json")
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Dataset JSON saved to {dataset_json_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to nnUNet format.",
        epilog="""Example usage:
  python dataset_to_nnunet.py --medvlsm_path /data/medvlsm --dataset_name busi --dataset_number 4

This script supports the following datasets:
- kvasirseg
- cvc
- bkai
- busi

This script will:
1. Preprocess the dataset if needed.
2. Match the JSON files and create a new dataset JSON.
3. Convert the dataset into nnUNet format.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--medvlsm_path', type=str, required=True,
                        help="Path to the medvlsm directory (e.g., '/data/medvlsm').")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., 'kvasirseg', 'cvc', 'bkai', 'busi').")
    parser.add_argument('--dataset_number', type=int, required=True,
                        help="Dataset number for nnUNet format (e.g., '1' for Dataset001).")

    args = parser.parse_args()

    # Process the dataset
    data_path = None
    if args.dataset_name.lower() == "kvasirseg":
        data_path = process_kvasirseg()
    elif args.dataset_name.lower() == "cvc":
        data_path = process_cvc()
    elif args.dataset_name.lower() == "bkai":
        data_path = process_bkai()
    elif args.dataset_name.lower() == "busi":
        data_path = process_busi()
    else:
        print(f"No processing function defined for dataset '{args.dataset_name}'.")
        sys.exit(1)

    # Match JSON files and create new dataset JSON
    dataset_json_matching(args.dataset_name, args.medvlsm_path, data_path, args.dataset_number)

    # Convert to nnUNet format
    to_nnunet_format(args.dataset_name, data_path, args.dataset_number)

if __name__ == "__main__":
    main()

"""
python dataset_to_nnunet.py \
--medvlsm_path /data/medvlsm \
--dataset_name kvasirseg \
--dataset_number 1 \


--zip_path /data/open_dataset/kvasir-seg.zip \

python dataset_to_nnunet.py \
    --medvlsm_path /data/medvlsm \
    --dataset_name cvc \
    --dataset_number 2 \
"""