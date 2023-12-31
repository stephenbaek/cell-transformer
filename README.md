# Neuronal Cell Body Segmentation using Swin Transformer

This repository contains the official implementation of [Mohammad Shafkat Islam, Pratyush Suryavanshi, Samuel M. Baule, Joseph Glykys, & Stephen Baek. (2023). A Deep Learning Approach for Neuronal Cell Body Segmentation in Neurons Expressing GCaMP Using a Swin Transformer. eNeuro, 10(9):ENEURO.0148-23.2023.](https://www.eneuro.org/content/10/9/ENEURO.0148-23.2023)

## Getting Started

### Install Anaconda and create conda environment:
```bash
conda env create -f swin_gpu_machine.yml
conda activate swin
mim install mmcv-full==1.5.0
cd ~/swin
pip install -v -e .
```
### Running Swin Transformer on Dataset:
1. Create a new folder inside the ~/swin/data/NEW_FOLDER. The naming convention should preferably be in a format similar to the current folders. 
baseline_05082021-014: fluorophoretype_date-xml
2. Inside the folder create Z-Projection folders. Put all the images from stacks in these Z-Projection folder. The naming convention should be similar to the current format. 
Stack_21_to_31_ZSeries-05082021-014.xml - C=0.tif: Stack_STARTSTACK_to_ENDSTACK_ZSeries- date-xml.xml - C=0.tif
You can remove all other folders from `~/swin/data` if you do not need the roi zip for those images.
3. Enter the following commands on the terminal:
```bash
python aws_cpu_scripts/1_run_python.py 0.05
```
Change the threshold parameter between [0.01 - 1] to obtain predictions higher then the set threshold
4. The swin transformer predicted roi zip should be in the ~/swin/predictions folder.

## Acknowledgement
This work was funded by NIH/NINDS R01NS115800 and the Iowa Neuroscience Institute. This research was partly supported by the computational resources provided by the University of Iowa, Iowa City, Iowa, and by The University of Iowa Hawkeye Intellectual and Developmental Disabilities Research Center (HAWK-IDDRC) P50 HD103556.

This work is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

## Contributors
- [Mohammad Shafkat Islam](https://datascience.virginia.edu/people/mohammad-islam), University of Virginia
-  Pratyush Suryavanshi, University of Iowa
-  Samuel Baule, University of Iowa
- [Stephen Baek](http://www.stephenbaek.com), University of Virginia
- [Joseph Glykys](https://medicine.uiowa.edu/pediatrics/profile/joseph-glykys), University of Iowa

## Citation
To cite this work, please use the following information:
```
@article {IslamENEURO.0148-23.2023,
	author = {Mohammad Shafkat Islam and Pratyush Suryavanshi and Samuel M. Baule and Joseph Glykys and Stephen Baek},
	title = {A Deep Learning Approach for Neuronal Cell Body Segmentation in Neurons Expressing {GCaMP} Using a Swin Transformer},
	journal = {{eNeuro}},
	volume = {10},
	number = {9},
	year = {2023},
	doi = {10.1523/ENEURO.0148-23.2023},
	URL = {https://www.eneuro.org/content/10/9/ENEURO.0148-23.2023}
}
```
