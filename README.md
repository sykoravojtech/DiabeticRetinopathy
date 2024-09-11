# KAGGLE: DiabeticRetinopathy
https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview

Create env
```
mamba create --prefix /storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy/env python=3.8 -y

qsub -I -l select=1:ncpus=16:ngpus=1:mem=40gb:scratch_ssd=20gb -l walltime=12:00:00

cd /storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy
export TMPDIR=$SCRATCHDIR
module add mambaforge
mamba activate /storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy/env

scp nademvit@onyx.metacentrum.cz:/storage/brno12-cerit/home/nademvit/kaggle_vs/diabeticRetinopathy/baseline/submission.csv .
```
libraries ìnside env
```
kaggle
Pillow
pandas
torch
torchvision
efficientnet_pytorch
sklearn
``

download data
```
export KAGGLE_CONFIG_DIR=/storage/brno12-cerit/home/nademvit/.kaggle/
kaggle competitions download -c diabetic-retinopathy-detection
kaggle competitions submit -c diabetic-retinopathy-detection -f submission.csv -m "Message"
```

https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy

```
cat train.zip.* > combined_train.zip
unzip combined_train.zip
```

# SCORES
- Baseline (B3, 120x120 image resolution):
* Val score: 0.43

+ increase image resoloution
Res 500x500 0.75
Res 728x728
