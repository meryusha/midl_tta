# Official Implementation of "Test-Time Adaptation for Combating Missing Modalities in Egocentric Videos"
This repository contains the official implementation of our paper, **[Test-Time Adaptation for Combating Missing Modalities in Egocentric Videos](https://openreview.net/pdf?id=1L52bHEL5d)**, accepted at ICLR 2025.

## 📜 Paper Summary
Understanding videos that contain multiple modalities is crucial, especially in egocentric videos, where combining various sensory inputs significantly improves tasks like action recognition and moment localization. However, real-world applications often face challenges with incomplete modalities due to privacy concerns, efficiency needs, or hardware issues. Current methods, while effective, often necessitate retraining the model entirely to handle missing modalities, making them computationally intensive, particularly with large training datasets. In this study, we propose a novel approach to address this issue at test time without requiring retraining. We frame the problem as a test-time adaptation task, where the model adjusts to the available unlabeled data at test time. Our method,**MiDl(Mutual information with self-Distillation)**, encourages the model to be insensitive to the specific modality source present during testing by minimizing the mutual information between the prediction and the available modality. Additionally, we incorporate self-distillation to maintain the model's original performance when both modalities are available. MiDl represents the first self-supervised, online solution for handling missing modalities exclusively at test time. Through experiments with various pretrained models and datasets, MiDl demonstrates substantial performance improvement without the need for retraining.

## 🔧 Repository Overview
This repo provides:
- ✅ PyTorch implementation of our proposed TTA method
- ✅ Evaluation scripts
- ✅ Pretrained models and datasets
- ✅ Reproducibility instructions

## 🚀 Getting Started  

Instructions for Running TTA Inference
## 1. **Data**

We provide the sharded videos and pre-trained checkpoints for you.

### **Steps to Prepare the Data**
- Download the necessary files using the links below.
- Unzip the folders and place them in this working repository.

After setup, your working repository should include the following folders:
- **EPIC_shards** [[Download Link]](https://drive.google.com/file/d/1vER03j1dBvLTEzMRTlvf_dRXqYTJFSvd/view?usp=sharing)  
- **EPIC_sounds_shards** [[Download Link]](https://drive.google.com/file/d/1qpBX8xhwXSC-E00cKLIlJFc-Eg3rPD3o/view?usp=sharing)  
- **checkpoints** [[Download Link]](https://drive.google.com/file/d/1XP8JgzjnE2thgqYE61IM5AnNwsXIKCmh/view?usp=sharing)  

**Note:**  
If you prefer not to download the full dataset, you can still test the code by downloading only the **checkpoints** folder. It contains all prediction files, enabling you to run evaluation code directly on them.

---

### **How We Prepared the Videos**
You can follow these steps if you wish to reproduce the process:

#### **a. Downloading Videos and Annotations**
- Download the **EPIC-KITCHENS** and **EPIC-SOUNDS** datasets:
  - Use the official [EPIC-KITCHENS download script](https://github.com/epic-kitchens/epic-kitchens-download-scripts).
- Annotations:
  - [EPIC-KITCHENS Annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
  - [EPIC-SOUNDS Annotations](https://github.com/epic-kitchens/epic-sounds-annotations)

#### **b. Video Pre-processing: Trimming and Sharding**
The datasets include long, untrimmed videos. Use the following scripts to trim and shard the videos into shorter clips based on annotated segments:

- Trimming:
  - `development_scripts/trimming/epic/trim_epic.py` (for EPIC-KITCHENS)
  - `development_scripts/trimming/epic/trim_epic_sound.py` (for EPIC-SOUNDS)

---

## 2. **Environment Setup**

1. Install the environment using the provided YAML file:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate midl_tta
   ```
---

## 3. **Running TTA Inference with MiDL**

#### **EPIC-SOUNDS**
Run the inference for EPIC-SOUNDS with the following command:
```bash
bash scripts/inference/tta_sounds/tta_inference_epic_sound.sh
```

#### **EPIC-KITCHENS**
Run the inference for EPIC-KITCHENS with the following command:
```bash
bash scripts/inference/tta_epic/tta_inference_epic.sh
```

### **Modifications**
- Update the **`PROP`** variable to set the missing ratio (`0.0`, `0.25`, `0.5`, `0.75`, or `1.0`).
- **Do not update** the **`SEED`** variable.

**Note:**  
We have included prediction files, so you don’t need to run inference to view results. If you wish to re-run the TTA process, **delete or rename the corresponding method folder** (e.g., for MiDL on EPIC-KITCHENS, delete or rename `checkpoints/EPIC-KITCHENS/midl`). Then re-run the appropriate script.

---

## 4. **Running TTA Inference with Baseline Methods**

### **Supported Baseline Methods**
- TENT
- SHOT
- ETA

#### **EPIC-SOUNDS**
Run the inference for EPIC-SOUNDS with the following command:
```bash
bash scripts/inference/tta_sounds/tta_inference_epic_sound_baselines.sh
```

#### **EPIC-KITCHENS**
Run the inference for EPIC-KITCHENS with the following command:
```bash
bash scripts/inference/tta_epic/tta_inference_epic_tta_baselines.sh
```

### **Modifications**
- Update the **`METHOD`** variable to select the TTA method (`shot-im`, `tent`, or `eta`).
- Update the **`PROP`** variable to set the missing ratio (`0.0`, `0.25`, `0.5`, `0.75`, or `1.0`).
- **Do not update** the **`SEED`** variable.


---

