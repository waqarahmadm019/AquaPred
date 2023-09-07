# Aqueous Solubility Predictor
This is the repo for the paper [Attention-based Graph Neural Network for Molecular Solubility Prediction](https://pubs.acs.org/doi/10.1021/acsomega.2c06702)
## Abstract
Drug discovery research is aimed at the discovery of new medications. Solubility
is an important physicochemical property in drug development. Active pharmaceutical ingredients (APIs) are essential substances for high drug efficacy. During DD
research, aqueous solubility (AS) is a key physicochemical attribute, required for API
characterization. High-precision in-silico solubility prediction reduces the experimental cost and time of drug development. Several artificial tools have been employed
for solubility prediction using machine learning and deep learning techniques. This
study aims to create different deep learning models that can predict the solubility
of a wide range of molecules using the largest currently available solubility dataset.
Simplified molecular-input line-entry system (SMILES) strings were used as molecular representation, models developed using simple graph convolution (SGConv), graph isomorphism network (GIN), graph attention network (GAT), and AttentiveFP network. Based on the performance of the models, the AttentiveFP-based network model
was finally selected. The model was trained and tested on 9943 compounds. The
model outperformed on 62 anticancer compounds with metric Pearson correlation R2
and root mean square error (RMSE) values of 0.52 and 0.61, respectively. AS can be
improved by graph algorithm improvement or more molecular properties addition.

## Model Flow Diagram
![ModelFlowDiagram123](https://user-images.githubusercontent.com/8627287/202886059-3830a1e8-01c0-4b30-a7a1-51eef4f1b986.jpg)


## Datasets
You can download the training and testing datasets from 
https://github.com/waqarahmadm019/AquaPred/tree/main/datasets

## Usage
1. Clone this repository
```
git clone https://github.com/waqarahmadm019/AquaPred
cd AquaPred
```

2. Setup Environment
```
conda create -n solubility --python=3.8 
conda activate solubility 
conda install -c pytorch pytorch=1.10.1
conda install -c conda-forge pytorch_geometric=2.0.3 
conda install -c conda-forge rdkit 
```

## test model TestModel.py
python TestModel.py

This code will use testset_novel.csv dataset
