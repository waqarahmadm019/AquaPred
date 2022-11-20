# Aqueous Solubility Predictor
This is the repo for the paper 'Attention-based Graph Neural Network for Molecular Solubility Prediction'
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

## How to download dataset and tool
Install anaconda and create conda environment<br>
Install pytorch, pytorch_geometric and rdkit.<br>
Run https://github.com/waqarahmadm019/AquaPred.git <br>
The above command will download the datasets and code automatically.<br>


![ModelFlowDiagram123](https://user-images.githubusercontent.com/8627287/202886059-3830a1e8-01c0-4b30-a7a1-51eef4f1b986.jpg)
## Steps how to create folds.
from folds import foldscreator<br>
traindf = pd.read_csv('New_fold/solubility.csv')<br>
foldscreator.createFoldsData(n_splits, 'smiles', 'y')<br>
## Model training. GraphNetDriver.py
train_loader   = DataLoader(train_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)<br>
test_loader  = DataLoader(test_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)<br>
model = AttentionConvNet().to(device)<br>
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)<br>
train(model, optimizer,train_loader)<br>
test(test_loader, model)<br>
## test model TestModel.py
model_file_name = 'saved_models/wholetrainmodel_6.model'.<br>
_,test_rmse, true, prediction = predicting(noveltest_loader, model).<br>

