import numpy as np
import torch.nn as nn
import json
import torch
import os

import Domain_Plotter as plt
from Utilities import loader_handler as lc
from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import dataset_reader as dr
from Architectures import Models





#######################################################
#************ USER INPUTS:                 ***********#
#######################################################

# Model Aspects
model_name              = "Simple_Test_myCPU_SDG"#"Model_Javier_Data_JavierSpherePacks" # The desired model name, avoid overwritting previous models
num_scales              = 4                                     # MS-Net paper use 4.

# Data aspects
NN_dataset_folder       = "../NN_Datasets/"
dataset_train_name      = "JavierSantos_FinneySpherePack_11092025_1Pressure.pt"
dataset_validation_name = "JavierSantos_Bentheimer_11092025_1Pressure.pt"
max_samples             = None      # Total samples loaded: None (default)=All
batch_size              = None      # Group size of train samples that influence one update on weights

# Learning aspects
N_epochs                = 50        # Number of times that all the samples are visited
learning_rate           = 0.00005     # Originally 0.001
loss_functions  = {
    "MS_MSE":                       {"obj": lf.MultiScaleLoss(nn.MSELoss()),                                            "Thresholded": False},
    "MS_MSE_VarNorm_inVoid":        {"obj": lf.MultiScaleLoss(lf.Mask_LossFunction(nn.MSELoss()), norm_mode='var'),     "Thresholded": False},
    "MS_MeanOutputError_inVoid":    {"obj": lf.MultiScaleLoss(lf.Mask_LossFunction(lf.MeanOutputError())),              "Thresholded": False},
    "MS_MSE_inVoid":                {"obj": lf.MultiScaleLoss(lf.Mask_LossFunction(nn.MSELoss())),                      "Thresholded": False},
    "MS_MeanPixelWiseRelativeError":{"obj": lf.MultiScaleLoss(lf.Mask_LossFunction(lf.MeanPixelWiseRelativeError())),   "Thresholded": False},
}

earlyStopping_loss      = "MS_MeanOutputError_inVoid"   # Which listed loss_function is used to stop trainning
backPropagation_loss    = "MS_MeanPixelWiseRelativeError"  # Which listed loss_function is used to calculate weights
optimizer               = 'ADAM'                        # One of: 'ADAM' or 'SGD' 
device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed                    = 42

#######################################################
#************ LOADING CONFIGS ************************#
#######################################################

    
NN_results_folder       = nnt.create_training_data_folder(base_dir="../NN_Results")
NN_model_weights_folder = NN_results_folder+"NN_Model_Weights/"
model_full_name         = NN_model_weights_folder+model_name
dataset_train_full_name = NN_dataset_folder+dataset_train_name
dataset_valid_full_name = NN_dataset_folder+dataset_validation_name
print(f"Optimizing with: {backPropagation_loss} for {optimizer}")
print()
print("Folder created for results: ",NN_results_folder)
print("Saving model weights in:    ",NN_model_weights_folder)
print("Model base name:            ",model_name)
print("Trainning with dataset:     ",dataset_train_full_name)
print("Validating with dataset:    ",dataset_valid_full_name)
print()
print("Max Samples:                ",max_samples)
print("Batch_size:                 ",batch_size)
print("N_epochs:                   ",N_epochs)
print("learning_rate:              ",learning_rate)
print("optimizer:                  ",optimizer)
print("device:                     ",device)
print("seed:                       ",seed)
print("earlyStopping_loss:         ",earlyStopping_loss)
print("backPropagation_loss:       ",backPropagation_loss)
print()

#######################################################
#************ SAVE METADATA ************************#
#######################################################

# Defina o caminho onde o arquivo será salvo
metadata_file = os.path.join(NN_results_folder, "metadata.txt")

# Monte o conteúdo do metadata
metadata_content = f"""
================= TRAINING METADATA =================

Model Aspects:
- model_name: {model_name}
- num_scales: {num_scales}

Data Aspects:
- NN_dataset_folder: {NN_dataset_folder}
- dataset_train_name: {dataset_train_name}
- dataset_validation_name: {dataset_validation_name}
- max_samples: {max_samples}
- batch_size: {batch_size}

Learning Aspects:
- N_epochs: {N_epochs}
- learning_rate: {learning_rate}
- optimizer: {optimizer}
- earlyStopping_loss: {earlyStopping_loss}
- backPropagation_loss: {backPropagation_loss}

Loss Functions:
{json.dumps({k: {"Thresholded": v["Thresholded"], "obj": str(v["obj"])} for k,v in loss_functions.items()}, indent=4)}

Paths:
- NN_results_folder: {NN_results_folder}
- NN_model_weights_folder: {NN_model_weights_folder}
- model_full_name: {model_full_name}
- dataset_train_full_name: {dataset_train_full_name}
- dataset_valid_full_name: {dataset_valid_full_name}

======================================================
"""
# Escreve no txt
with open(metadata_file, "w") as f:
    f.write(metadata_content)

print(f"Metadata saved at: {metadata_file}")
#######################################################
#************ LOADING DATA          ******************#
#######################################################
# Set seed to random initializations
nnt.set_seed(seed) # If empty (None) make it random, if integer eliminates randomness

print("Loading Trainning Data ... ")
# Load Multiscale Dataset for trainning
scaled_train_data   = torch.load(dataset_train_full_name,  weights_only=False)
loader_train        = dr.MultiScaleDataset.get_dataloader(scaled_train_data, batch_size)
loader_train        = loader_train._cut(max_samples)
del scaled_train_data

print("Loading Validation Data ... ")
# Load Multiscale Dataset for validation
scaled_valid_data   = torch.load(dataset_valid_full_name,  weights_only=False)
loader_valid        = dr.MultiScaleDataset.get_dataloader(scaled_valid_data, batch_size)
loader_valid        = loader_valid._cut(max_samples)
del scaled_valid_data


#######################################################
#******************** MODEL **************************#
#######################################################
print("Loading Model ... ")
model = Models.MS_Net(
    num_scales    = num_scales,     # num of trainable convNets
    num_features  = 1,              # input features (Euclidean distance)
    num_filters   = 2,              # num of kernels on each layer of the finest model (most expensive)
)

#######################################################
#************ COMPUTATIONS ***************************#
#######################################################
print(f"Starting Train on {device}... \n")

#### MODEL TRAINNING


if optimizer == 'ADAM': 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
else:
    raise Exception(f"Optimizer {optimizer} is not implemented.")
    
nnt.full_train(
    model,
    loader_train.loader,
    loader_valid.loader,                    
    loss_functions,                                      
    earlyStopping_loss,
    backPropagation_loss,
    optimizer,
    N_epochs=N_epochs,
    weights_file_name=model_full_name,
    results_folder=NN_results_folder,
    device=device,
    )

print("Ending Train ... ")

### DELETE MODEL AFTER USING IT
mh.delete_model(model)
del loader_train
del loader_valid