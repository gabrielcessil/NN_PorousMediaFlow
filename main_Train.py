import numpy as np
import torch.nn as nn
import json
import torch

import Domain_Plotter as plt
from Utilities import loader_handler as lc
from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh

from network import MS_Net


#######################################################
#************ USER INPUTS:                 ***********#
#######################################################

# Model Aspects
model_name      = "Model_Javier_Data_JavierSpherePacks" # The desired model name, avoid overwritting previous models
num_scales      = 4                                     # MS-Net paper use 4.

# Data aspects
dataset_name    = "JavierSantos_FinneySpherePack.pt"
max_samples     = None  # Total samples loaded: None (default)=All
batch_size      = 1     # Group size of train samples that influence one update on weights
val_ratio       = None  # Fraction of max_samples used to validate (None=no splitting)
train_ratio     = None  # Fraction of max_samples used to train (None=no splitting)

# Learning aspects
N_epochs        = 50    # Number of times that all the samples are visited
learning_rate   = 0.0001 # Originally 0.001
loss_functions  = {
    "MS_MSE":           {"obj": lf.MultiScaleLoss(nn.MSELoss()),                    "Thresholded": False},
    "MS_MSE_VarNorm":   {"obj": lf.MultiScaleLoss(nn.MSELoss(), norm_mode='var'),   "Thresholded": False},
    "MS_PixelWise":     {"obj": lf.MultiScaleLoss(lf.PixelWisePercentualError()),   "Thresholded": False},
    "Mean_output":      {"obj": lf.MultiScaleLoss(lf.MeanOutputError()),            "Thresholded": False}
}
earlyStopping_loss      = "MS_MSE_VarNorm" # Which listed loss_function is used to stop trainning
backPropagation_loss    = "MS_MSE_VarNorm" # Which listed loss_function is used to calculate weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################
#************ LOADING CONFIGS ************************#
#######################################################

# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
NN_dataset_folder       = config_loaded["NN_dataset_folder"]
NN_model_weights_folder = config_loaded["NN_model_weights_folder"]
NN_results              = config_loaded["NN_results"]
model_full_name         = NN_model_weights_folder+model_name
dataset_full_name       = NN_dataset_folder+dataset_name

#######################################################
#************ LOADING DATA          ******************#
#######################################################
print("Loading Data ... ")
# Load Multiscale Dataset
scaled_data             = torch.load(dataset_full_name,  weights_only=False)


# Divide list of pairs into array of N inputs and array of N targets
scaled_input_tensors    = []
scaled_output_tensors   = []
for input_scales, target_scales in scaled_data:
    scaled_input_tensors.append(input_scales)
    scaled_output_tensors.append(target_scales)
del scaled_data
dataloader = lc.Data_Loader(scaled_input_tensors, scaled_output_tensors, batch_size)
del scaled_input_tensors
del scaled_output_tensors

# Divide data into train / validation / test (TO BE REPLACED)
if train_ratio is not None and val_ratio is not None :
    train_batch_loader, val_loader, test_loader = dataloader.get_splitted(
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
        batch_size  = batch_size,
        max_samples = max_samples)
else:
    dataloader = dataloader._cut(max_samples)
    train_batch_loader, val_loader, test_loader = dataloader.loader, dataloader.loader, dataloader.loader
    

dataloader.print_stats(train_batch_loader, val_loader, test_loader)
del dataloader

#######################################################
#******************** MODEL **************************#
#######################################################
print("Loading Model ... ")

model = MS_Net(
    num_scales   := num_scales,     # num of trainable convNets
    num_features  = 1,              # input features (Euclidean distance)
    num_filters   = 2,              # num of kernels on each layer of the finest model (most expensive)
)

#######################################################
#************ COMPUTATIONS ***************************#
#######################################################
print("Starting Train ... ")

#### MODEL TRAINNING
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

nnt.full_train(
    model,
    train_batch_loader,
    val_loader,                    
    loss_functions,                                      
    earlyStopping_loss,
    backPropagation_loss,
    optimizer,
    N_epochs=N_epochs,
    weights_file_name=model_full_name,
    results_folder=NN_results,
    device=device
    )

### DELETE MODEL AFTER USING IT
mh.delete_model(model)
del train_batch_loader
del val_loader
del test_loader
