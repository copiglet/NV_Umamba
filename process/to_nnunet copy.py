#!/usr/bin/env python3

import os
import json
import shutil
import sys
from PIL import Image
import numpy as np

def to_nnunet_format(dataset_name, data_path, dataset_number):
    """
    Converts the dataset into nnUNet format by copying images and masks to the appropriate directories.

    Parameters:
    - dataset_name (str): Name of the dataset (e.g., 'kvasirseg', 'cvc', 'bkai', 'busi')
    - data_path (str or dict): Path to the dataset files (e.g., unzipped directory or dict with paths)
    - dataset_number (int): Dataset number for nnUNet format.
    """
    dataset_id = str(dataset_number).zfill(3)
    nnunet_dataset_dir = os.path.join(
        "..",
        f"data/nnUNet_raw/Dataset{dataset_id}_{dataset_name}"
    )
    dataset_json_path = os.path.join(nnunet_dataset_dir, "dataset.json")

    # Load the dataset JSON
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)

    # Define paths
    imagesTr_dir = os.path.join(nnunet_dataset_dir, "imagesTr")
    labelsTr_dir = os.path.join(nnunet_dataset_dir, "labelsTr")
    imagesTs_dir = os.path.join(nnunet_dataset_dir, "imagesTs")
    labelsTs_dir = os.path.join(nnunet_dataset_dir, "labelsTs")

    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    os.makedirs(imagesTs_dir, exist_ok=True)
    os.makedirs(labelsTs_dir, exist_ok=True)

    # Function to process entries (training or test)
    def process_entries(entries, images_dest_dir, labels_dest_dir):
        for entry in entries:
            image_name = entry["image_name"]  # e.g., 'kvasirseg001_0000'
            ori_img_name = entry["ori_img_name"]
            ori_mask_name = entry.get("ori_mask_name", ori_img_name)
            image_ext = os.path.splitext(ori_img_name)[1]  # Get the original file extension

            # Remove '_0000' from image_name to get label_name
            label_name = image_name.replace('_0000', '')

            if dataset_name.lower() == "busi":
                class_dir = entry["class_dir"]
                # Source paths
                src_image_path = os.path.join(class_dir, ori_img_name)
                # Destination paths
                dst_image_path = os.path.join(images_dest_dir, image_name + image_ext)
                # Copy image
                shutil.copyfile(src_image_path, dst_image_path)

                # Process masks
                # Find all mask files corresponding to the image
                base_mask_name = os.path.splitext(ori_img_name)[0] + "_mask"
                mask_files = [f for f in os.listdir(class_dir) if f.startswith(base_mask_name)]
                if not mask_files:
                    print(f"No mask files found for image {ori_img_name} in {class_dir}")
                    continue

                # Initialize combined mask
                combined_mask = None

                # Load and combine masks
                for mask_file in mask_files:
                    mask_path = os.path.join(class_dir, mask_file)
                    mask_image = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask_image)
                    if combined_mask is None:
                        combined_mask = mask_array
                    else:
                        combined_mask = np.maximum(combined_mask, mask_array)

                # Save combined mask
                combined_mask_image = Image.fromarray(combined_mask)
                dst_mask_path = os.path.join(labels_dest_dir, label_name + image_ext)
                combined_mask_image.save(dst_mask_path)
            else:
                # Dataset-specific images and masks directories
                if dataset_name.lower() == "kvasirseg":
                    images_dir = os.path.join(data_path, "Kvasir-SEG", "images")
                    masks_dir = os.path.join(data_path, "Kvasir-SEG", "masks")
                elif dataset_name.lower() == "cvc":
                    images_dir = data_path["images_dir"]
                    masks_dir = data_path["masks_dir"]
                elif dataset_name.lower() == "bkai":
                    images_dir = data_path["images_dir"]
                    masks_dir = data_path["masks_dir"]
                else:
                    print(f"Dataset '{dataset_name}' not recognized.")
                    sys.exit(1)

                # Source paths
                src_image_path = os.path.join(images_dir, ori_img_name)
                src_mask_path = os.path.join(masks_dir, ori_mask_name)

                # Destination paths
                dst_image_path = os.path.join(images_dest_dir, image_name + image_ext)
                dst_mask_path = os.path.join(labels_dest_dir, label_name + image_ext)  # Use label_name here

                # Copy files
                shutil.copyfile(src_image_path, dst_image_path)
                shutil.copyfile(src_mask_path, dst_mask_path)

    # Process training entries
    process_entries(
        dataset_json["imagesTr"],
        imagesTr_dir,
        labelsTr_dir
    )

    # Process test entries
    process_entries(
        dataset_json["imagesTs"],
        imagesTs_dir,
        labelsTs_dir
    )

    print(f"Data converted to nnUNet format in {nnunet_dataset_dir}")
