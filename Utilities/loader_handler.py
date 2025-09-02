import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

no_collate=lambda batch: batch

def compute_loader_outputs(model, loader, N_Samples, batch_size=1, shuffle=False):
    model.eval()
    subset      = Subset(loader.dataset, range(N_Samples))    
    new_loader  = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

    outputs = []
    with torch.no_grad():
        for batch_inputs, batch_targets in new_loader:
            batch_output = model(batch_inputs)
            outputs.append(batch_output)

    return outputs, new_loader

class Data_Loader(Dataset):
    
    def __init__(self, inputs_tensor, outputs_tensor, batch_size, shuffle=False):
        
        self.inputs, self.outputs = inputs_tensor, outputs_tensor
        self.batch_size = batch_size if batch_size is not None else len(self)
        self.loader = DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)  # Automatically creates a DataLoader
        
    
    def get_splitted(self, train_ratio, val_ratio, max_samples=None, collate=None):
        train_dataset, val_dataset, test_dataset = self._split(
            train_ratio=train_ratio, val_ratio=val_ratio, max_samples=max_samples
        )
        x, y = train_dataset[0]
        
        if collate is None: collate=no_collate
        
        train_loader    = DataLoader(train_dataset, batch_size=min(max_samples, self.batch_size), collate_fn=collate, shuffle=True)
        val_loader      = DataLoader(val_dataset, batch_size=max_samples, collate_fn=collate, shuffle=False)    # No shuffle for validation
        test_loader     = DataLoader(test_dataset, batch_size=max_samples,  collate_fn=collate, shuffle=False)  # No shuffle for test
        
        return train_loader, val_loader, test_loader

    def print_stats(self, train_loader, val_loader, test_loader):
        
        total_samples = (
            len(train_loader.dataset) +
            len(val_loader.dataset) +
            len(test_loader.dataset)
        )
        # First batch for shape inspection
        train_first_batch = next(iter(train_loader))
        if len(train_first_batch) != 2: raise Exception(f"- Loader has {len(train_first_batch)} items and must have 2 (inputs and targets separately).")

        batch_input, batch_output = train_first_batch

        # Handle multi-scale or single-scale input
        if isinstance(batch_input, list):
            input_shape = [x.shape for x in batch_input]
            batch_size = batch_input[0].shape[0]
        elif isinstance(batch_input, torch.Tensor):
            input_shape = batch_input.shape
            batch_size = batch_input.shape[0]
        else:
            raise Exception("Batched input must be either a torch.Tensor or a list of torch.Tensors (multi-scale).")
    
        output_shape = batch_output.shape if isinstance(batch_output, torch.Tensor) else [y.shape for y in batch_output]
        
        
        
        # Print overall dataset/batch info
        print("=== Dataloader Summary ===")

        print(f"- Total samples considered: {total_samples}")
        print(f"  -- Train samples     : {len(train_loader.dataset)}")
        print(f"  -- Validation samples: {len(val_loader.dataset)}")
        print(f"  -- Test samples      : {len(test_loader.dataset)}")
        
        print(f"- Number of training batches: {len(train_loader)}")
        print("- Batch shape details: (batch_size, channels, depth, height, width)")
        print(f"  -- Train batch input shape : {input_shape}")
        print(f"  -- Train batch output shape: {output_shape}")
    
        # === Sanity checks ===
        print("=== Dataloader Sanity Check ===")
        val_first_batch = next(iter(val_loader))
        test_first_batch = next(iter(test_loader))
    
        val_input, val_output = val_first_batch
        test_input, test_output = test_first_batch
    
        if isinstance(val_input, list):
            val_input_shape = [x.shape for x in val_input]
            val_batch_size = val_input[0].shape[0]
        else:
            val_input_shape = val_input.shape
            val_batch_size = val_input.shape[0]
    
        if isinstance(test_input, list):
            test_input_shape = [x.shape for x in test_input]
            test_batch_size = test_input[0].shape[0]
        else:
            test_input_shape = test_input.shape
            test_batch_size = test_input.shape[0]
    
        # Compare batch counts
        if len(train_loader) == len(val_loader) == len(test_loader):
            print("- All loaders have the same number of batches.")
        else:
            raise Exception("Number of batches differ across loaders.")
    
        # Compare input shapes
        if input_shape == val_input_shape == test_input_shape:
            print("- Input shapes are consistent across loaders.")
        else:
            raise Exception("Input shapes differ across loaders.")
    
        # Compare output shapes
        if isinstance(output_shape, list):
            val_output_shape = [y.shape for y in val_output]
            test_output_shape = [y.shape for y in test_output]
        else:
            val_output_shape = val_output.shape
            test_output_shape = test_output.shape
    
        if output_shape == val_output_shape == test_output_shape:
            print("- Output shapes are consistent across loaders.")
        else:
            raise Exception("Output shapes differ across loaders.")
    
        # Compare batch sizes
        if batch_size == val_batch_size == test_batch_size:
            print(f"- Batch size is consistent across loaders: {batch_size}")
        else:
            raise Exception("Batch sizes differ across loaders.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    def _split(self, train_ratio=0.7, val_ratio=0.15, max_samples=None):
        total_size = len(self)
        if total_size <= 3:
            raise Exception("TensorsDataset must have at least 3 samples in order to split")
    
    
        # Compute sizes ensuring at least 1 sample per dataset
        train_size = max(1, int(train_ratio * total_size))
        val_size = max(1, int(val_ratio * total_size))
        
        # Ensure remaining samples go to the test set
        test_size = max(1, total_size - (train_size + val_size))  
    
        # Adjust if rounding errors cause an overflow
        if train_size + val_size + test_size > total_size:
            train_size = total_size - (val_size + test_size)
    
        # Generate indices and split dataset    
        train_dataset = self._cut(train_size, start=0)
        val_dataset   = self._cut(val_size, start=train_size)
        test_dataset  = self._cut(test_size, start=train_size + val_size)
        
        if train_size > max_samples:
            train_dataset = self._cut(max_samples)
        if val_size > max_samples:
            val_dataset   = self._cut(max_samples)
        if test_size > max_samples:
            test_dataset  = self._cut(max_samples)

        return train_dataset, val_dataset, test_dataset
    
    def _cut(self, size, start=0): 
        total_size = len(self.inputs)
        size = total_size if size is None else size
        end = start + size
        assert 0 <= start < total_size, "Invalid start index"
        assert 0 <= end <= total_size, "Subset range exceeds dataset size"
        new_inputs  = self.inputs[start:end]
        new_outputs = self.outputs[start:end]
        return Data_Loader(new_inputs, new_outputs, self.batch_size)
    
    
    