# VGN: Vascular graph network for ovarian OR-PAM

This repository contains the implementation of **VGN** from the manuscript:
**"Vascular graph network for ovarian lesion classification using optical-resolution photoacoustic microscopy"**

## Manuscript abstract:
Diagnosing ovarian lesions is challenging because of their heterogeneous clinical presentations. Some benign ovarian conditions, such as endometriosis, can have features that mimic cancer. We use optical-resolution photoacoustic microscopy (OR-PAM) to study the differences in ovarian vasculature between cancer and various benign conditions. In this study, we converted OR-PAM vascular data into vascular graphs augmented with physical vascular properties. From 94 ovarian specimens, a custom vascular graph network (VGN) was developed to classify each graph as either normal ovary, one of three benign pathologies, or cancer. We demonstrated for the first time that, by leveraging the intrinsic similarity between vascular networks and graph constructs, VGN provides stable predictions from sampling surface areas as small as 3 mm x 0.12 mm. In diagnosing cancer, VGN achieved 79.5% accuracy and an area under the receiver operating characteristic curve (AUC) of 0.877. Overall, VGN achieved a five-class classification accuracy of 73.4%.
**keywords**: photoacoustic microscopy, ovarian cancer, multiparametric imaging, deep learning, graph neural network

## Repository structure
- **`data/`** – Dataset utilities
- **`img_processing`** = Image processing utilities (preprocesing of raw RF data and postprocessing for PA feature extraction)
- **`models/`** – Model architectures (choice from various existing GNN structures and the final structure used in VGN)  
- **`train/`** – Training, evaluation, and testing scripts
- **`matlab_scripts`** - Postprocessing code was originally developed in MATLAB, python version is now available in **`img_processing`**
- **`artifacts/`** - Temp data
- **`docs/images/`** – Figures for model illustration and results

## Model architecture
<p align="center">
  <img src="docs/images/graph_generation.png" alt="Procedure for generating a vascular graph from the radiofrequency data of a B scan." width="600"/>
</p>

<p align="center">
  <img src="docs/images/model_structure.png" alt="VGN architecture. A vascular graph is generated from 41 consecutive B scans. Solid arrows and lines indicate data flow and connections inside the model. The two message-passing mechanisms employed in VGN are described in the boxes below the network structure diagram. Purple letters indicate the model’s trainable parameters." width="600"/>
</p>

## Model performance

<p align="center">
  <img src="docs/images/model_performance.png" alt="VGN classification performance for five-class classification and the corresponding average one-class-versus-rest ROC curve for differentiating each of the five classes. BC: benign cystic, BS: benign solid, Ems: endometriosis." width="600"/>
</p>

## Model interpretation

<p align="center">
  <img src="docs/images/model_interp.png" alt="Visualizing VGN model predictions on representative C scans of different ovarian lesions. (a) normal ovary. (b) benign cystic lesion. (c) benign solid lesion. (d) endometriosis. (e) cancer. BC: benign cystic, BS: benign solid, Ems: endometriosis." width="600"/>
</p>


## Getting started
```bash
git clone https://github.com/YXLin1159/PAM-VGN-ovarian-cancer.git
cd PAM-VGN-ovarian-cancer
conda env create -f environment.yml
conda activate pam-vgn