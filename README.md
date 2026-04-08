# VBTPlan: automatic segmentation of CT scans for vaginal bracytherapy planning

The lack of robust automatic tools for segmentation in vaginal brachytherapy (VBT) limits efficiency and reproducibility in clinical practice. We aimed to develop a framework for automatic segmentation of the clinical target volume (CTV) and organs-at-risk (OARs) for endometrial carcer patients that undergo vaginal cuff brachytherapy.  
__Authors:__ Adrià Casamitjana \
__Contact email:__ adria.casamitjana@udg.edu

## Contents

### 1.- General description
We developed a three-step framework that capitalised on nnUNet and adapted it to our context to segment the CTV and OARs on pre-treatment computed tomography (CT) scans. Our method was adaptive to different contouring protocols used in clinical practice, where images were partly labelled, by either providing labels within a region of interest and/or considering only a subset of present structures. A dataset of 289 patients treated between 2014-2021 was used for model development (139/35/115 for training, validation and testing).  
 Casamitjana et al., Robust framework for adaptive field-of-view automatic segmentation in vaginal brachytherapy for 
 endometrial cancer (2026). (submitted)

### 2.- Installation
To run this package you just need python3 and virtualenv installed in your system. For a sanity check, you could run
``` 
which python3
which virtualenv
```
and check that the result is pointing towards an existing file in your system (and it's not empty).

To install the VBTPlan package, please run the following steps:
* Move into your preferred directory: ```cd /path/to/your/directory```
* Clone the GitHub repository: ```git clone https://github.com/acasamitjana/VBTPlan.git```
* Install python environment and the libraries required
```
virtualenv --p python3 .venv 
source .venv/bin/activate
pip install pyproject.toml
```

### 3.- Usage

#### 3.1.- Inference
Open a local terminal:
* Change directory to the project root. 
* Activate your local environment
* Then execute the inference.py file
```
cd /path/to/this/software
source .venv/bin/activate
vbtplan --i input_file --o output_dir 
```
The script has the following parameters:
* __Input file (--i):__ path to input DICOM or NIFTI file
* __Output file (--o):__ path to output directory
* __CPU device flag (--cpu):__ if force run on CPU (instead of GPU)

This will populate the output directory with the output at different steps of the pipeline
The final RT struct can be found at the _dicom_ directory and the NIFTI file at the _final_ directory (both within the output directory)

#### 3.2.- Inference with other state-of-the-art methods
We outsource different pre-trained models (reported in our manuscript). The inference scripts can be found at: scripts/sota_methods:
1. SegResNet: _inference_segresnet.py_
2. UNet-R: _inference_unetr.py_
3. Swint-UNet-R: _inference_swin_unetr.py_

All three scripts have the same input parameters as the main program: input file, output file, CPU flag.

