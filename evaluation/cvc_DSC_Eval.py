# -*- coding: utf-8 -*-
"""
Evaluation script for CVC dataset (0: background, 1: polyp).
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_path',
    type=str,
    default='',
    help='Path to the ground truth label files.'
)
parser.add_argument(
    '--seg_path',
    type=str,
    default='',
    help='Path to the predicted segmentation files.'
)
parser.add_argument(
    '--save_path',
    type=str,
    default='',
    help='Path to save the evaluation results.'
)

args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(os.path.join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = []
seg_metrics['Polyp_DSC'] = []

def compute_dice_coefficient(truth, prediction):
    """
    Computes the Dice coefficient between the ground truth and the prediction.
    """
    intersection = np.sum((truth == 1) & (prediction == 1))
    denominator = np.sum(truth == 1) + np.sum(prediction == 1)
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    # Load ground truth and segmentation
    gt_nii = nb.load(os.path.join(gt_path, name))
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(os.path.join(seg_path, name)).get_fdata())

    # Compute Dice coefficient for polyp (label 1)
    if np.sum(gt_data == 1) == 0 and np.sum(seg_data == 1) == 0:
        DSC = 1.0
    elif np.sum(gt_data == 1) == 0 and np.sum(seg_data == 1) > 0:
        DSC = 0.0
    else:
        DSC = compute_dice_coefficient(gt_data == 1, seg_data == 1)
    seg_metrics['Polyp_DSC'].append(round(DSC, 4))

# Save the results to a CSV file
dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)

# Compute average DSC
average_dsc = dataframe['Polyp_DSC'].mean()
print(20 * '>')
print(f'Average DSC for {os.path.basename(seg_path)}: {average_dsc}')
print(20 * '<')

"""

python cvc_DSC_Eval.py \
    --gt_path /data/U-Mamba/data/nnUNet_raw/Dataset002_cvc/labelsTs \
    --seg_path /data/U-Mamba/data/nnUNet_raw/Dataset002_cvc/outputTs \
    --save_path /data/U-Mamba/data/nnUNet_raw/Dataset002_cvc/metric_DSC_umambabot.csv

""" 