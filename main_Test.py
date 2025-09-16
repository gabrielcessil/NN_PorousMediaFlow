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

NN_results              = "../NN_Results/"
#NN_model_weights_folder = "NN_Trainning_12_September_2025_07:46AM/NN_Model_Weights/"
#NN_model_weights_folder = "NN_Trainning_12_September_2025_06:59PM/NN_Model_Weights/"
NN_model_weights_folder = "NN_Trainning_16_September_2025_11:05AM/NN_Model_Weights/"

model_name              = "Simple_Test_myCPU_SDG_ProgressTracking_50.pth"

NN_dataset_folder       = "../NN_Datasets/"
dataset_name            = "JavierSantos_FinneySpherePack_11092025_1Pressure.pt"
examples_shape          = [256, 256, 256]
device                  = 'cpu'

#######################################################
#************ LOADING CONFIGS       ******************#
#######################################################
model_path              = NN_results+NN_model_weights_folder
model_full_name         = model_path+model_name
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

outputs, loader = lc.compute_loader_outputs(model, dataloader.loader)


#######################################################
#************ TESTING               ******************#
#######################################################
        
#*********** ONE EXAMPLE SAMPLE ANALYSIS *************#
# Select sample to be analyzed: first sample
ra.analyze_input_target_output_domain(loader, outputs, model_path)

#*********** PERFORMANCE OF EACH SAMPLE *************#
#ra.analyze_domain_error(loader, outputs, model_path)

#*********** STATISTICAL PERFOMANCE      *************#
#ra.analyze_population_distributions(loader, outputs, model_path)


del outputs
del dataloader
