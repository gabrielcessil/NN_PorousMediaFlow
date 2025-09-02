import h5py
from scipy.io import loadmat
import numpy as np
from scipy.ndimage import distance_transform_edt
import Domain_Plotter as plt
from scipy.io import loadmat
import os
import torch


def sanity_check(rock, target, solid_default_value=0):
    # SANITY CHECK: NO VELOCITY IN ROCK SOLID
    solid_mask = rock==solid_default_value
    return np.mean(abs(target[solid_mask]))==0

def get_mat_file_name_pairs(directory):
    # Store full paths (without .mat) keyed by identifier
    solids = {}
    vfields = {}

    for file in os.listdir(directory):
        if file.endswith('.mat'):
            no_ext = os.path.splitext(file)[0]
            full_path = os.path.join(directory, no_ext)

            if '_solid' in file:
                key = no_ext.replace('_solid', '')
                solids[key] = full_path
            elif '_vfield' in file:
                key = no_ext.replace('_vfield', '')
                vfields[key] = full_path

    # Create pairs where both solid and vfield exist
    common_keys = set(solids.keys()) & set(vfields.keys())
    pairs = [(solids[k], vfields[k]) for k in sorted(common_keys)]

    return pairs


def get_SamplePair(example_filename, simu_filename):
    velocity_data = loadmat(simu_filename)
    solid_data = loadmat(example_filename)

    rock_input = solid_data['solid'] # Originally 1-solid, 0-void
    velocity = velocity_data["vfield"]
    
    rock_dist_trans = distance_transform_edt(rock_input == 1.0)
    
    example_output_ux, example_output_uy, example_output_uz = np.split(velocity, 3, axis=-1)
    example_output_ux = np.squeeze(example_output_ux, axis=-1)
    example_output_uy = np.squeeze(example_output_uy, axis=-1)
    example_output_uz = np.squeeze(example_output_uz, axis=-1)
    
    if not sanity_check(rock_input, example_output_uz, solid_default_value=0.0): 
        raise Exception("Target and Rock domains do not match.")
        
    rock_dist_trans = rock_dist_trans/10
    example_output_uz = example_output_uz*10e2
    
    return rock_input, rock_dist_trans, example_output_uz 

def get_Tensors(filespath):
    mat_files = get_mat_file_name_pairs(filespath)

    rock_inputs = []
    rock_dist_transforms = []
    examples_output_uz = []

    for rock_filename, simu_filename in mat_files:
        rock_input, rock_dist_trans, example_output_uz = get_SamplePair(rock_filename, simu_filename)
        
        rock_inputs.append([rock_input])
        rock_dist_transforms.append([rock_dist_trans])
        examples_output_uz.append([example_output_uz])
        
    # Convert to tensors
    rock_input_tensor = torch.tensor(np.array(rock_inputs, dtype=np.float32))
    example_input_tensor = torch.tensor(np.array(rock_dist_transforms, dtype=np.float32))
    example_target_tensor = torch.tensor(np.array(examples_output_uz, dtype=np.float32))

    return rock_input_tensor, example_input_tensor, example_target_tensor # (B, C, H, W, D)

def get_Arrays(filespath):
    mat_files = get_mat_file_name_pairs(filespath)

    rock_inputs = []
    rock_dist_transforms = []
    examples_output_uz = []

    for rock_filename, simu_filename in mat_files:
        rock_input, rock_dist_trans, example_output_uz = get_SamplePair(rock_filename, simu_filename)
        
        rock_inputs.append(rock_input)
        rock_dist_transforms.append(rock_dist_trans)
        examples_output_uz.append(example_output_uz)

    return rock_inputs, rock_dist_transforms, examples_output_uz

        
