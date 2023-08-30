# A Deep Learning Approach for Neuronal Cell Body Segmentation in neurons expressing GCaMP using a Swin Transformer

This repository contains the codes of neuronal cell body detection and segmentation using Swin transformer. 

## Introduction

The different steps to install the Swin transformer are described below:

### Install Anaconda and create conda environment:
```bash
conda env create -f swin_ui.yml
conda activate swin
mim install mmcv-full==1.5.0
cd ~/swin
pip install -v -e .
```
### Running Swin Transformer on Dataset:
1. Create a new folder inside the ~/swin/data/NEW_FOLDER. The naming convention should preferably be in the format similar to the current folders. 
baseline_05082021-014: fluorophoretype_date-xml

2. Inside the folder create Z-Projection folder. Put all the images from stacks in these Z-Projection folder. The naming convention should be like current format. 
Stack_21_to_31_ZSeries-05082021-014.xml - C=0.tif: Stack_STARTSTACK_to_ENDSTACK_ZSeries- date-xml.xml - C=0.tif
You can remove all other folders from ~/swin/data if you do not need the roi zip for those images.

3. Enter the following commands on the terminal:
```bash
python aws_cpu_scripts/1_run_python.py 0.05
```
Change the threshold parameter between [0.01 - 1] to obtain predictions higher then the set threshold

4. The swin transformer predicted roi zip should be in the ~/swin/predictions folder.
