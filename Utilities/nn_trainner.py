import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os.path
import json
from Utilities import usage_metrics as um
from Utilities import plotter
import time


# valid_loss_functions: dict of loss functions, where the key identify them
# train_loss_function: must be the name of the loss function used as key inside valid_loss_functions
def full_train(model, 
               train_batch_loader,
               valid_loader,
               loss_functions,
               earlyStopping_loss,
               backPropagation_loss,
               optimizer,
               scheduler=None,
               N_epochs=1000,
               weights_file_name = "model_weights",
               results_folder = "",
               device="cpu",
               saving_dec_perc=10):
        
    # Make sure to use same device
    model.to(device)
    
    # Initialize tracking
    average_computation = 0
    train_costs_history = []
    val_costs_history   = []
    best_valid_loss     = np.inf    
    best_model_path     = weights_file_name+"_LowerValidationLoss.pth"
    model_paths         = []
    progress_points     = set(int(N_epochs * i / 100) for i in range(0, 101, 1))  
    best_model          = None
    
    print("\n\nTrainning session has started.\n")
    # Trainning process
    for epoch_index in range(N_epochs):
        
        epoch_timestamp_start = time.perf_counter()
        
        # Learn updating model
        model = train_one_epoch( 
            model               = model,
            train_batch_loader  = train_batch_loader,
            loss_function       = loss_functions[backPropagation_loss], 
            optimizer           = optimizer,
            scheduler           = scheduler,
            device              = device)
        
        # Get learning metrics
        train_avg_loss, valid_avg_loss = validate_one_epoch(
                                              model         = model, 
                                              train_loader  = train_batch_loader,
                                              valid_loader  = valid_loader,
                                              loss_functions= loss_functions,
                                              device        = device)
        
        train_costs_history.append(train_avg_loss)
        val_costs_history.append(valid_avg_loss)
        epoch_timestamp_stop    = time.perf_counter()
        diff                    = (epoch_timestamp_stop - epoch_timestamp_start)
        average_computation     += (diff - average_computation) / (epoch_index + 1)
        
        # Save tracking based on best performance. If it is monotonic: will save every step (disc access)
        
        if valid_avg_loss[earlyStopping_loss] < best_valid_loss:
            percent = round((1-(valid_avg_loss[earlyStopping_loss] / best_valid_loss)) * 100, 2)  # Calcula o percentual relativo
            print(f"New best solution achieved at {epoch_index} / {N_epochs} ({percent:.1f}% better)")
            best_valid_loss = valid_avg_loss[earlyStopping_loss]
            best_model      = model.state_dict() # Temporarilly saves best model (RAM)
            

        # Save tracking based on % of epochs
        if epoch_index in progress_points:
            percent = round((epoch_index / N_epochs) * 100, 2)  # Calcula o percentual relativo
            
            print(f"\nExecuting epoch {epoch_index} / {N_epochs} ({percent:.1f}%)")
            print("--> Allocated memory {} (MB) ".format(round(um.get_memory_usage())))
            print("--> Average epoch processing time (seconds): ", round(average_computation,6))
            
            model_path = weights_file_name+"_ProgressTracking_{}.pth".format(round((epoch_index / N_epochs) * 100))
            torch.save(model.state_dict(), model_path)
            model_paths.append(model_path)
            Plot_LossHistory(train_costs_history, val_costs_history, output_path=f"{results_folder}LossHistory.png")
     
    # Saves on disc the best model (once after trainning)
    model_paths.append(best_model_path)
    torch.save(best_model, best_model_path)
    
    """
    # Save training metadata
    metadata = {
        "Timestamp": timestamp,
        "Best Validation Loss": best_valid_loss,
        "Early Stopping Loss": earlyStopping_loss,
        "Backpropagation Loss": backPropagation_loss,
        "Total Epochs": N_epochs,
        "Batch Size": len(next(iter(train_batch_loader))),
        "Device": device,
        "Optimizer": optimizer.__class__.__name__ if not isinstance(optimizer, list) else [optimizer_i.__class__.__name__ for optimizer_i in optimizer] ,
        "Learning Rate": optimizer.param_groups[0]['lr'] if not isinstance(optimizer, list) else [optimizer_i.param_groups[0]['lr'] for optimizer_i in optimizer],
        "Model Details":  model.metadata if hasattr(model, 'metadata') else ""
    }
    save_training_metadata(f"{results_folder}{timestamp}/training_metadata", metadata)
    """
    return best_model, model_paths

# Move tensors to device
# If obj is a list, move each element to device
def move_to_device(obj, device):
    if isinstance(obj, (list, tuple)):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj.to(device)
    
def train_one_epoch(model, train_batch_loader, loss_function, optimizer, scheduler, device='cpu'):
    model.train()
    
    # For each batch
    for batch_inputs, batch_targets in train_batch_loader:
        
        # Make he batch
        # If the batch input is list (multi-scale)
        if isinstance(batch_inputs, list) or isinstance(batch_inputs, torch.Tensor):
            # Load batch data on device
            batch_inputs    = move_to_device(batch_inputs,  device)
            batch_targets   = move_to_device(batch_targets, device)
            
            optimizer.zero_grad() # Reinicia os gradientes para calculo do passo dos pesos
            batch_outputs   = model(batch_inputs) # Calcula Saídas
            loss = loss_function["obj"](batch_outputs, batch_targets) # Calcula Custo
            loss.backward() # Calcula gradiente em relacao ao Custo
            optimizer.step()
        
        else: raise Exception(f"The train batch loader must be one of torch.Tensor of list but got type {type(batch_inputs)}.")
            
        if not scheduler is None: scheduler.step() # Realiza um passo no learning rate
            
    return  model

def validate_one_epoch(model, train_loader, valid_loader, loss_functions, device):
    return get_loader_loss(model, train_loader, loss_functions, device), get_loader_loss(model, valid_loader, loss_functions, device)
    

def get_loader_loss(model, loader, loss_functions, device):
    
    results = {loss_name: 0.0 for loss_name in loss_functions}
    
    with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
        
      model.eval() # Entre em modo avaliacao, desabilitando funcoes exclusivas de treinamento (ex:Dropout)
      
      # For each batch listed in loader
      for batch_inputs, batch_targets in loader:
        
        # If batch's samples are listed: iterate the list
        if not (isinstance(batch_inputs, list) or isinstance(batch_inputs, torch.Tensor)): 
            raise Exception(f"The train batch loader must be one of torch.Tensor of list but got type {type(batch_inputs)}.")
            
        # Load batch data on device
        batch_inputs = move_to_device(batch_inputs, device)
        batch_targets = move_to_device(batch_targets, device)
        
        # For each loss function listed
        for loss_name, loss_function in loss_functions.items():
            
            # If the loss function is in thresholded mode: use predict mode to get output
            if loss_function["Thresholded"]: batch_outputs = model.predict(batch_inputs)
            # If the loss function is NOT in thresholded mode: use forward mode to get output
            else: batch_outputs = model(batch_inputs)
            
            # Compute loss based on output
            aux  = loss_function["obj"](batch_outputs, batch_targets).item()
            results[loss_name] += aux
    
    return results


 


    
def get_example(loader, i_batch, i_example):
    input_example, target_example = next(itertools.islice(loader, i_batch, None)) 
    input_example, target_example = input_example[i_example], target_example[i_example]
    
    print(f"  Input shape: {input_example.shape}")  # Shape of the in_shapeinput
    print(f"  Output shape: {target_example.shape}\n")  # Shape of the output

    return input_example, target_example

#######################################################
#********************* PLOTTERS **********************#
#######################################################

import matplotlib.pyplot as plt
import numpy as np

def Plot_LossHistory(train_cost_history, val_cost_history, normalize=False, output_path=None):
    
    # Extract training history
    train_history_dicts = {loss_name: [] for loss_name in train_cost_history[0]}
    for epoch_dict in train_cost_history:
        for loss_name in epoch_dict:
            train_history_dicts[loss_name].append(epoch_dict[loss_name])
    
    # Extract validation history
    valid_history_dicts = {loss_name: [] for loss_name in val_cost_history[0]}
    for epoch_dict in val_cost_history:
        for loss_name in epoch_dict:
            valid_history_dicts[loss_name].append(epoch_dict[loss_name])
    
    
    
    num_plots = len(train_history_dicts)
    fig, axes = plt.subplots(num_plots, 3, figsize=(18, 5 * num_plots))
    if num_plots == 1:
        axes = np.array([axes])
        
    def scale(data):
        return data / np.max(data)

    for idx, loss_name in enumerate(train_history_dicts.keys()):
        
        train_loss_history = train_history_dicts[loss_name]
        valid_loss_history = valid_history_dicts[loss_name]
        
        if normalize:
            train_loss_history = scale(train_loss_history)
            valid_loss_history = scale(valid_loss_history)
        
        # Plot 1: Full cost history
        x = range(len(train_cost_history))
        axes[idx, 0].plot(x, train_loss_history, label='train', color='blue')
        axes[idx, 0].plot(x, valid_loss_history, label='validation', color='red')
        axes[idx, 0].set_xlabel('Epochs')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].set_title(f'Cost History ({loss_name})')
        axes[idx, 0].legend()
        axes[idx, 0].set_ylim(bottom=0.0)

        # Plot 2: Average loss of the last 10%
        if len(train_loss_history) > 10: # Ensure there's enough data
            axes[idx, 1].plot(x[-10:], train_loss_history[-10:], label='train', color='blue')
            axes[idx, 1].plot(x[-10:], valid_loss_history[-10:], label='validation', color='red')
        else:
            axes[idx, 1].plot(x[-len(train_loss_history):], train_loss_history[-10:], label='train', color='blue')
            axes[idx, 1].plot(x[-len(train_loss_history):], valid_loss_history[-10:], label='validation', color='red')
            
        axes[idx, 1].set_xlabel('Epochs')
        axes[idx, 1].set_ylabel('Loss')
        axes[idx, 1].set_title(f'Recent Cost History ({loss_name})')
        axes[idx, 1].legend()
        axes[idx, 1].set_ylim(bottom=0.0)

        # Plot 3: Loss difference history
        loss_difference = np.array(valid_loss_history) - np.array(train_loss_history)
        axes[idx, 2].plot(x, loss_difference, label='Validation - Train', color='black')
        axes[idx, 2].set_xlabel('Epochs')
        axes[idx, 2].set_ylabel('Validation Loss - Train Loss')
        axes[idx, 2].set_title(f'Loss difference History ({loss_name})')
        axes[idx, 2].legend()
        axes[idx, 2].set_ylim(bottom=0.0)

    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
def Run_Example(model, loader, loss_functions, main_loss, title="Example", i_batch=0, i_example=0):
    
    input_example, target_example = get_example(loader, i_batch, i_example)
    
    batch_example_input = input_example.unsqueeze(0) # Add batch dimension in 3D tensor for forward
    pred_example        = model.predict(batch_example_input) # Disable graphs 
    
    pred_example        = pred_example.squeeze(0) # Remove batch dimension from predictions (1,C,H,W)->(C,H,W)
    loss_example        = loss_functions[main_loss]["obj"](pred_example, target_example) # Compute the example of loss
    print(f"{title},  {main_loss} Loss: {loss_example}")
    
    #plotter.display_image_tensor(input_example, title+"_input")
    #plotter.display_image_tensor(target_example, title+"_target")
    #plotter.display_image_tensor(pred_example, title+"_pred")
    plotter.display_example(input_example, pred_example, target_example, title=title)

def Plot_DataloaderBatch(dataloader, num_images=5):

    data_iter = iter(dataloader)
    images, masks, img_names, mask_names = next(data_iter)  # Get filenames too
    
    largura_base = 10  # Largura base da figura
    incremento_por_imagem = 3  # Incremento na largura por imagem
    largura_total = largura_base + incremento_por_imagem * num_images
    altura = 6  # Altura fixa da figura
    fig, axes = plt.subplots(2, num_images, figsize=(largura_total, altura))


    for i in range(num_images):  # Display num_images samples
        img = np.transpose(images[i].numpy(), (1, 2, 0))  # Convert tensor format
        mask = masks[i].squeeze().numpy()  # Remove extra dimensions

        axes[0, i].imshow(img)
        axes[0, i].set_title(img_names[i])  # Set title with image filename
        axes[0, i].axis("off")

        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(mask_names[i])  # Set title with mask filename
        axes[1, i].axis("off")

    plt.show()
    
    
#######################################################
#************ METADATA HANDLERS **********************#
#######################################################

def save_training_metadata(filename, data):

    # Ensure the filename has the .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    # Check if the file exists
    if os.path.isfile(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)  # Load existing data
    else:
        existing_data = []

    # Append new metadata
    existing_data.append(data)

    # Save back to JSON
    with open(filename, "w+") as file:
        json.dump(existing_data, file, indent=4)
        
def save_metadata(filename, data, header=None):

    file_exists = os.path.isfile(filename)

    if file_exists:
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])

    df.to_csv(filename, index=False, header=header)
