import json
import torch
import numpy as np
import Domain_Plotter as plt
from Utilities import loader_handler as lc
from Utilities import result_analyzer as ra
from network import MS_Net
from network_tools import get_masks
from torch.utils.data import DataLoader, Subset
from Utilities import array_handler as ah
from Utilities import dataset_reader as dr
    
    
#######################################################
#************ INPUTS                ******************#
#######################################################

#model_name          = "Model_Javier_Data_JavierSpherePacks_LowerValidationLoss.pth" # The desiresd model name, avoid overwritting previous models
model_name          = "Model_Javier_Data_JavierSpherePacks_ProgressTracking_33__GPU_29082025.pth"
dataset_name        = "JavierSantos_FinneySpherePack.pt"
device              = 'cpu'

#######################################################
#************ LOADING CONFIGS       ******************#
#######################################################
# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
examples_shape          = config_loaded["Rock_shape"]
NN_dataset_folder       = config_loaded["NN_dataset_folder"]
NN_model_weights_folder = config_loaded["NN_model_weights_folder"]
NN_results              = config_loaded["NN_results"]
model_full_name         = NN_model_weights_folder+model_name
dataset_full_name       = NN_dataset_folder+dataset_name

#######################################################
#************ LOADING MODEL         ******************#
#######################################################
# Make sure to create same structure
model = MS_Net(
    num_scales   := 4,   # num of trainable convNets
    num_features  = 1,   # input features (Euclidean distance)
    num_filters   = 2,   # num of kernels on each layer of the finest model (most expensive)
)
model.load_state_dict(torch.load(model_full_name, map_location=torch.device(device)))
model.eval()


#######################################################
#************ LOADING ROCK EXAMPLE *******************#
#######################################################

scaled_data = torch.load(dataset_full_name,  weights_only=False, map_location=torch.device(device))
dataloader  = dr.MultiScaleDataset.get_dataloader(scaled_data, batch_size=1)
del scaled_data

outputs, loader = lc.compute_loader_outputs(model, dataloader.loader, N_Samples=10)


#######################################################
#************ TESTING               ******************#
#######################################################
        
#*********** ONE EXAMPLE SAMPLE ANALYSIS *************#
# Select sample to be analyzed: first sample
ra.analyze_domain_sample_data(loader, outputs)

#*********** PERFORMANCE OF EACH SAMPLE *************#
ra.analyze_domain_error(loader, outputs)

#*********** STATISTICAL PERFOMANCE      *************#
#ra.analyze_population(loader, outputs)

#ra.sanity_check(loader, outputs)

del outputs
del dataloader