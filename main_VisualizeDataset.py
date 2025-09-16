import numpy as np
import json
import torch
import matplotlib.pyplot as plt

import Domain_Plotter as pl
from Utilities import dataset_reader as dr

#######################################################
#************ INPUTS                ******************#
#######################################################

NN_dataset_folder       = "../NN_Datasets/"
dataset_name            = "JavierSantos_FinneySpherePack_11092025_1Pressure.pt"
#dataset_name            = "JavierSantos_Bentheimer_11092025_1Pressure.pt"

examples_shape          = [256, 256, 256]
device                  = 'cpu'

#######################################################
#************ LOADING CONFIGS       ******************#
#######################################################
dataset_full_name       = NN_dataset_folder+dataset_name

scaled_data = torch.load(dataset_full_name,  weights_only=False, map_location=torch.device(device))
dataloader  = dr.MultiScaleDataset.get_dataloader(scaled_data, batch_size=1)
del scaled_data

target_means = {}
input_means = {}


with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
  # For each batch listed in loader
  for b_i, (batch_inputs, batch_targets) in enumerate(dataloader.loader):
      for i, (input_sample, target_sample) in enumerate( zip(batch_inputs,batch_targets)):
          input_sample  = input_sample.squeeze(0).squeeze(0).numpy()
          target_sample = target_sample.squeeze(0).squeeze(0).numpy()
          dmx,dmy,dmz   = target_sample.shape
          
          datasetFolder = dataset_name.removesuffix(".pt")
          shape =input_sample.shape
          #pl.Plot_Domain(input_sample, filename=f"input_sample_{b_i}_{shape}", remove_value=[])
          #pl.Plot_Domain(target_sample, filename=f"target_sample_{b_i}_{shape}", remove_value=[])
          pl.Plot_Domain((input_sample!=0).astype(np.uint8),   filename=f"{NN_dataset_folder}{datasetFolder}/{b_i}_{shape}_solid_sample", remove_value=[1])
          pl.Plot_Domain((target_sample!=0).astype(np.uint8),  filename=f"{NN_dataset_folder}{datasetFolder}/{b_i}_{shape}_solid_sample_fromTarget", remove_value=[1])
          pl.Plot_Continuous_Domain_2D(input_sample[-1,:,:],    filename=f"{NN_dataset_folder}{datasetFolder}/{b_i}_{shape}_input_sample_slice")
          pl.Plot_Continuous_Domain_2D(target_sample[-1,:,:],   filename=f"{NN_dataset_folder}{datasetFolder}/{b_i}_{shape}_target_sample_slice")
          
          if dmx not in target_means: target_means[dmx] = []
          if dmx not in input_means: input_means[dmx]   = []
          target_means[dmx].append(np.mean(target_sample[input_sample>0]))
          input_means[dmx].append(np.mean(input_sample[input_sample>0]))
    
plt.figure(figsize=(8, 8),dpi=300)  # bigger, nicer figure

for scale, (dmx, means_dmx) in enumerate(target_means.items()):
    array_scales = np.full(len(means_dmx), scale)
    array_means = np.array(means_dmx)

    plt.scatter(
        array_scales,
        array_means,
        label=str(dmx),
        alpha=0.7,             # transparency
        s=80,                  # point size
        edgecolors="black",    # black outline
        linewidths=0.5
    )

plt.yscale("log")  # log scale on Y
plt.xlabel("Scale", fontsize=22)
plt.ylabel("Mean Values (log scale)", fontsize=22)
plt.title("Target Means per Scale", fontsize=22, fontweight="bold")

plt.grid(True, which="both", ls="--", alpha=0.5)  # major + minor grid
plt.legend(title="DMX", fontsize=14)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 8),dpi=300)  # bigger, nicer figure

for scale, (dmx, means_dmx) in enumerate(input_means.items()):
    array_scales = np.full(len(means_dmx), scale)
    array_means = np.array(means_dmx)

    plt.scatter(
        array_scales,
        array_means,
        label=str(dmx),
        alpha=0.7,             # transparency
        s=80,                  # point size
        edgecolors="black",    # black outline
        linewidths=0.5
    )

plt.yscale("log")  # log scale on Y
plt.xlabel("Scale", fontsize=22)
plt.ylabel("Mean Values (log scale)", fontsize=22)
plt.title("Input Means per Scale", fontsize=22, fontweight="bold")

plt.grid(True, which="both", ls="--", alpha=0.5)  # major + minor grid
plt.legend(title="DMX", fontsize=14)
plt.tight_layout()
plt.show()
      
      #pl.plot_scatter_sampled(, np.array(target_means))
          
          