import numpy as np
import Domain_Plotter as plt
import json
import torch

from Utilities import loader_creator as lc

dataset_name    = "JavierSantos_FinneySpherePack.pt"

with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
NN_dataset_folder       = config_loaded["NN_dataset_folder"]
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
loader = lc.Data_Loader(scaled_input_tensors, scaled_output_tensors, 1)
del scaled_input_tensors
del scaled_output_tensors

with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
  # For each batch listed in loader
  for b_i, (batch_inputs, batch_targets) in enumerate(loader):
      for i, (input_sample, target_sample) in enumerate( zip(batch_inputs,batch_targets)):
          input_sample  = input_sample.squeeze(0).numpy()
          target_sample = target_sample.squeeze(0).numpy()
          
          plt.Plot_Domain(input_sample, filename=f"input_sample_{b_i}_{i}", remove_value=[1])
          plt.Plot_Domain(target_sample, filename=f"target_sample_{b_i}_{i}", remove_value=[1])
#######################################################
#******************** MODEL **************************#
#######################################################