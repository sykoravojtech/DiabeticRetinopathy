# Diabetic Retinopathy Detection

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Diabetic%20Retinopathy%20Detection-blue)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview)

## Overview

This repository provides a minimal baseline for the [Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview) competition on Kaggle. The goal is to classify retinal fundus images according to the severity of diabetic retinopathy.

## Environment Setup

Use **Mamba** to create and manage your Python environment:

```bash
mamba create --prefix ./env python=3.8 -y
mamba activate ./env
```

Install the necessary packages:

```bash
pip install \
    kaggle \
    Pillow \
    pandas \
    torch \
    torchvision \
    efficientnet-pytorch \
    scikit-learn
```

## Data Acquisition

Ensure your Kaggle API token is placed under `~/.kaggle/kaggle.json` (or set `KAGGLE_CONFIG_DIR` accordingly):

```bash
export KAGGLE_CONFIG_DIR=~/.kaggle
kaggle competitions download -c diabetic-retinopathy-detection
```

Combine and unpack the training archives:

```bash
cat train.zip.* > train_combined.zip
unzip train_combined.zip
```

Submit your predictions:

```bash
kaggle competitions submit \
  -c diabetic-retinopathy-detection \
  -f submission.csv \
  -m "Your submission message"
```

## Baseline Results

- **Model**: EfficientNet-B3  
- **Image Resolution**: 120×120  
- **Validation Score**: 0.43  

> **Tip**: Increasing the input resolution yields a significant boost:
> - 500×500 → 0.75  
> - 728×728 → _TBA_

---

## Developer

All commands for running on the Metacentrum HPC “Onyx” cluster are provided below.

### Environment & Interactive Session

```bash
# 1. Create and activate the environment
mamba create --prefix /storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy/env python=3.8 -y

# 2. Start an interactive GPU session
qsub -I \
  -l select=1:ncpus=16:ngpus=1:mem=40gb:scratch_ssd=20gb \
  -l walltime=12:00:00

# 3. Navigate to project directory
cd /storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy

# 4. Use scratch for temporary files
export TMPDIR=$SCRATCHDIR

# 5. Load Mambaforge and activate the environment
module add mambaforge
mamba activate /storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy/env
```

### Data Download & Preparation

```bash
# Point to your Kaggle credentials
export KAGGLE_CONFIG_DIR=/storage/brno12-cerit/home/nademvit/.kaggle/

# Download the competition data
kaggle competitions download -c diabetic-retinopathy-detection

# Merge and unpack training archives
cat train.zip.* > combined_train.zip
unzip combined_train.zip
```

### Data Transfer

```bash
# Retrieve the latest submission file
scp nademvit@onyx.metacentrum.cz:/storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy/baseline/submission.csv .
```
