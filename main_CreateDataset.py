import torch
import torch.nn as nn
import json
import sys 
from torch.utils.data import DataLoader

from Utilities import dataset_reader as dr
from Utilities import loss_functions as lf
import Domain_Plotter as plt

import JavierData 
import DannyData

#######################################################
#************ USER INPUTS: Hyperparameters ***********#
#######################################################

# Model Aspects
dataset_name    = "JavierSantos_Bentheimer"
#datapath        = "/home/gabriel/Desktop/Dissertacao/Simulations/Javier/Javier_TrainningData/"
datapath        = "/home/gabriel/Desktop/Dissertacao/Simulations_Data/Javier/Javier_ValidationData/"

num_scales = 4

#######################################################
#************ LOADING CONFIGS ************************#
#######################################################

# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
NN_dataset_folder = config_loaded["NN_dataset_folder"]
dataset_full_name = NN_dataset_folder+dataset_name
images_full_name = NN_dataset_folder+"/"+dataset_name+"_Images/"




#######################################################
#************ LOADING ROCK EXAMPLE *******************#
#######################################################
# Use a custom class to handle different data sources
dataset = JavierData.Lazy_RockDataset(datapath)
print("Rock and Simulation collected.")

# Make a multiscale dataset for multiscale architecture
scaled_dataset = dr.MultiScaleDataset(dataset, num_scales=num_scales)
print("Dataset created.")

# Save dataset
response = input(f"Confirm that you are dealing with {datapath} to create dataset {dataset_name} [Y/N]: ").strip().lower()
if response not in ['yes', 'y', 'Y']:
    print("Operation cancelled.")
    sys.exit(1)

#######################################################
#************ VISUALIZATION **************************#
#######################################################

# Visualization (optional)
#dataloader = DataLoader(scaled_dataset, batch_size=1, shuffle=False)

buffer      = []
buffer_size = None

# Iterate MultiScale Dataset
for i in range(len(scaled_dataset)):
    print(f"\n Catching item {i}/{len(scaled_dataset)}")
    # GET DATA Path (for plotting and errors tracking only)
    name1, _    = dataset.paths[i]
    name1       = name1.strip("'").split('/')[-1].replace('.mat', '')    
    try:
        input_scales, target_scales = scaled_dataset[i]
        if input_scales is None or target_scales is None: print("Itens cant be None")
        # Analize the elements content: shape and variable type.
        if i==0:
            if isinstance(input_scales, list): print(f"\nStacking data with shapes {[inp.shape for inp in input_scales]}. Make sure each is either (C,H,W,D) or (C,H,W).")
            elif isinstance(input_scales,  torch.Tensor): print(f"\nStacking data with shape {input_scales.shape}. Make sure it is either (C,H,W,D) or (C,H,W).")
            else: raise Exception(f"Stacking variables of type {type(input_scales)}. Make sure it is either torch.Tensor or a list of torch.Tensor.")
   
        # MANAGE DATASET DATA
        # If the item is not available / valid, do not append it
        if input_scales is None or target_scales is None: continue
        # Append valid itens
        buffer.append((input_scales, target_scales))
        # Manage buffer (make sub-datasets, limiting the total of samples), if buffer_size is None do not create sub-datasets
        if buffer_size is not None and len(buffer) >= buffer_size:
            torch.save(buffer, f"{dataset_full_name}_part_{i//buffer_size}.pt")
            buffer = []
        
    except Exception as e:
        print(f"Error getting item {i} ({name1}) from dataset : {e}")




    """
    # VISUALIZATION
    # Get the highest resolution tensors from the end of each scale list
    input_tensor, target_tensor = input_scales[-1], target_scales[-1]
    # Each has shape: (1, 1, D, H, W) → remove batch & channel dims for plotting
    input_tensor                = input_tensor[-1].squeeze(0).squeeze(0)
    target_tensor               = target_tensor[-1].squeeze(0).squeeze(0) # Make it 
    input_np                    = input_tensor.detach().cpu().numpy()
    target_np                   = target_tensor.detach().cpu().numpy()
    plt.plot_heatmap(input_np[:, :, 0],   title=f"input_{name1}", output_file=f"{images_full_name}{name1}_input")
    plt.plot_heatmap(target_np[:, :, 0],  title=f"target_{name1}", output_file=f"{images_full_name}{name1}_target")
    plt.Plot_Domain((~(target_np == 0.0)).astype(float), filename=f"{images_full_name}{name1}_3Drock", remove_value=[1.0])
    plt.Plot_Domain(target_np, filename=f"{images_full_name}{name1}_3Dtarget")
    """
    
if buffer:
    torch.save(buffer, f"{dataset_full_name}.pt")