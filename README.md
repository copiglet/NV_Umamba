# U-Mamba: Biomedical Image Segmentation

Welcome to the official repository for **U-Mamba**, a deep learning model designed to enhance long-range dependencies in biomedical image segmentation.

📬 Stay updated! Join our [mailing list](https://forms.gle/bLxGb5SEpdLCUChQ7).

---

## 🔧 Installation

**Requirements:**
- **OS:** Ubuntu 20.04
- **CUDA:** 11.8

### Steps:
1. **Set up a virtual environment:**
   ```bash
   conda create -n umamba python=3.10 -y
   conda activate umamba
   ```
2. **Install PyTorch 2.0.1:**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Install Mamba:**
   ```bash
   pip install causal-conv1d>=1.2.0
   pip install mamba-ssm --no-cache-dir
   ```
4. **Download the repository:**
   ```bash
   git clone https://github.com/bowang-lab/U-Mamba
   cd U-Mamba/umamba
   pip install -e .
   ```
5. **Sanity test:**
   ```bash
   python
   >>> import torch
   >>> import mamba_ssm
   ```

![U-Mamba Network](https://github.com/bowang-lab/U-Mamba/blob/main/assets/U-Mamba-network.png)

---

## 🏋️ Model Training

### 📥 Dataset Setup
Download the dataset [here](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing) and place it in the `data` folder.

U-Mamba is built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). If you plan to use your own dataset, refer to the [nnU-Net dataset format guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

### 🔄 Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 🏋️ Train 2D Models
- **Train U-Mamba_Bot:**
  ```bash
  nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot
  ```
- **Train U-Mamba_Enc:**
  ```bash
  nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc
  ```

### 🏋️ Train 3D Models
- **Train U-Mamba_Bot:**
  ```bash
  nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot
  ```
- **Train U-Mamba_Enc:**
  ```bash
  nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc
  ```

---

## 🔍 Inference

- **Predict using U-Mamba_Bot:**
  ```bash
  nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta
  ```
- **Predict using U-Mamba_Enc:**
  ```bash
  nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaEnc --disable_tta
  ```

ℹ️ `CONFIGURATION` can be either `2d` or `3d_fullres`.

---

## ⚙️ Additional Notes

### 1️⃣ Path Customization
U-Mamba defaults to `U-Mamba/data`. You can modify data paths in `umamba/nnunetv2/path.py` as follows:
```python
base = '/home/user_name/Documents/U-Mamba/data'
nnUNet_raw = join(base, 'nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results')
```

### 2️⃣ AMP Issues
AMP might cause `nan` values in Mamba. A trainer without AMP is available [here](https://github.com/bowang-lab/U-Mamba/blob/main/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncNoAMP.py).

---

## 📄 Citation
If you use U-Mamba in your research, please cite:
```bibtex
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}
```

---

## 🙏 Acknowledgements
We appreciate the authors of the public datasets and the contributors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [Mamba](https://github.com/state-spaces/mamba) for making their code open-source.
