import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from Utilities import loader_handler as lc


#######################################################
#**** CUSTOMIZATION TO DEAL WITH MS-Net **************#
#######################################################

# Made by Gabriel Silveira
class MultiScaleDataset(Dataset):
    def __init__(self, base_dataset, num_scales=4):
        self.base_dataset = base_dataset
        self.num_scales = num_scales

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        input_tensor, target_tensor = self.base_dataset[idx]

        input_scales = self.get_coarsened_list(input_tensor, num_scales=self.num_scales)
        target_scales = self.get_coarsened_list(target_tensor, num_scales=self.num_scales)
        
        return input_scales, target_scales
    
    def get_coarsened_list(self, x, num_scales):    
        ds_x = []
        ds_x.append(x)
        for i in range( num_scales-1 ): 
            ds_x.append( self.scale_tensor( ds_x[-1], scale_factor=1/2 ) )
        return ds_x[::-1] # returns the reversed list (small images first)
        
    def add_dims(self, x, num_dims):
        for dims in range(num_dims):
            x = x[np.newaxis]
        return x
    
    def scale_tensor(self, x, scale_factor=1):
        
        # Downscale image
        if scale_factor<1:
            return nn.AvgPool3d(kernel_size = int(1/scale_factor))(x)
        
        # Upscale image (Never used in the code...)
        elif scale_factor>1:
            for repeat in range (0, int(np.log2(scale_factor)) ):  # number of repeatsx2
                for ax in range(2,5): # (B,C,  H,W,D), repeat only the 3D axis, not batch and channel
                    x=x.repeat_interleave(repeats=2, axis=ax)
            return x
        
        # No alter image
        elif scale_factor==1:
            return x
        
        else: raise ValueError(f"Scale factor not understood: {scale_factor}")
        

    @staticmethod
    def get_dataloader(scaled_data, batch_size, verbose=False):
        # Divide list of pairs into array of N inputs and array of N targets
        scaled_input_tensors    = []
        scaled_output_tensors   = []
        
        # Separate scaled_data in different lists 
        for input_i_scales, target_i_scales in scaled_data: # Each Input and Target is a list of Tensors(scales)
            scaled_input_tensors.append(input_i_scales)
            scaled_output_tensors.append(target_i_scales)
        dataloader = lc.Data_Loader(scaled_input_tensors, scaled_output_tensors, batch_size=batch_size)
        del scaled_input_tensors
        del scaled_output_tensors
        
        # Print dimensions
        if verbose:
            inputs, targets = next(iter(dataloader.loader))    # Get first sample item: List of Tensors (scales)
            print(f" Loader with {len(dataloader.loader)} samples")
            print(f" Multi-Scale samples: {len(inputs)}")
            for i,(scale_input, scale_target) in enumerate(zip(inputs, targets)):
                print(f" -- scale {i}: in_shape: {scale_input.shape}, target_shape: {scale_target.shape}")
            print()
        
        return dataloader
    
    
        
        
        
        
        
class LBM_Dataset_Processor:
    """
    Dataset Loader & Preprocessor

    Loads a `.pt` dataset, handling folder structures and various formats, ensuring 
    correct conversion to PyTorch tensors for model training.

    Features:
    - Reads `.pt` files and processes inputs/targets.
    - Reshapes arrays as needed and converts to tensors.
    - Handles lists, NumPy arrays, and different data structures.
    - Stores tensors as `self.inputs` and `self.targets`.

    Usage:
    ```
    processor = LBM_Dataset_Processor("your_dataset.pt", array_shape=(50, 50))
    inputs, targets = processor.get_tensors()
    ```
    """

    def __init__(self, file_path):
        """
        Initializes the dataset processor by loading and processing data.

        Args:
            file_path (str): Path to the `.pt` dataset file.
        """
        self.file_path = file_path
        
        self.inputs, self.targets = self.load_data()

    def array_to_numpy(self, variable):
        """Ensures the input is a NumPy array."""
        if isinstance(variable, np.ndarray):
            return variable
        elif isinstance(variable, list):
            return np.array(variable)
        raise TypeError("Dataset samples must be a list or a numpy ndarray.")

    def reshape_array(self, array):
        """Reshapes an array to the expected shape if necessary."""
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")
        if array.shape == self.array_shape:
            return array  # Already correct shape
        try:
            return array.reshape(self.array_shape)
        except ValueError:
            raise ValueError(f"Cannot reshape array of shape {array.shape} to {self.array_shape}")

    def convert_to_tensor(self, array, flatten=False, add_channel_dim=False):
        """Converts a NumPy array to a PyTorch tensor with optional modifications."""
        tensor = torch.tensor(array, dtype=torch.float32)
        if flatten:
            tensor = tensor.view(-1)  # Flatten
        if add_channel_dim:
            tensor = tensor.unsqueeze(0)  # Add channel dim (for Conv2D input)
        return tensor

    def load_data(self):
        """Loads and processes the dataset into tensors."""
        dataset = torch.load(self.file_path, weights_only=False)
        
        inputs = []
        targets = []
        
        # Convert pairs of arrays into tensors
        for pair in dataset:
            if len(pair) != 2:
                raise ValueError("Each dataset sample must contain 2 elements (input and ouput).")
            
            # Store tensors
            inputs.append(self.convert_to_tensor(pair[0]))
            targets.append(self.convert_to_tensor(pair[1]))  # Stack targets together

        # Convert lists to tensors
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        return inputs, targets

    def get_tensors(self):
        """Returns the processed input and output tensors."""
        return self.inputs, self.targets


    