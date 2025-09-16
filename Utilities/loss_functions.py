import torch.nn as nn
from torchmetrics.classification import Accuracy
from torch.nn import BCELoss, functional
import torch



#######################################################
#************ LOSS FUNCTION UTILITIES ****************#
#######################################################

# Apply threshold
class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return (torch.sigmoid(x) > self.threshold).float()
    
# Defines the specific pixels that the loss function might look into
class Mask_LossFunction(nn.Module):
    def __init__(self, lossFunction, mask_law=None):
        super(Mask_LossFunction, self).__init__()
        
        self.lossFunction = lossFunction
        
        if mask_law is None: 
            self.mask_law = self._default_mask_law
        else:
            self.mask_law = mask_law
            
    # Do not consider cells with 0 value  
    # The loss function used must be a mean across the tensor lenght, 
    # so that the quantity of solid cells do not affect the loss
    def _default_mask_law(self,output, target, threshold=0): 
        # Mask consider only target != 0, i.e, non-solid cells
        #mask = (target > threshold) | (target < -threshold)
        mask = torch.abs(target) > threshold
        return mask
    
    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        mask = self.mask_law(output, target)
        
        target = target[mask]
        output = output[mask]

        return self.lossFunction(output, target)
    
    
    
    
#######################################################
#************ LOSS FUNCTIONS  ************************#
#######################################################
    
class Custom_BCE(nn.Module):
    def __init__(self):
        super(Custom_BCE, self).__init__()
        self.bce = BCELoss()
        
    def forward(self,output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss_Accuracy forward: Tensors have different sizes ({output.size()} vs {target.size()})")

        # Ensure float type
        output = output.float()
        target = target.float()
        
        # Clamp output to avoid log(0) issues
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        
        return self.bce(output, target)
    
# MY COMPOSED FUNCTIONS
    
class Cosine(nn.Module):
    def __init__(self, eps=1e-8):
        super(Cosine, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")

        # Flattened versions for angle loss
        T_flat = target.flatten(start_dim=1).float()  # [B, D]
        P_flat = output.flatten(start_dim=1).float()  # [B, D]
        
        # Cosine similarity per sample
        T_norm = T_flat.norm(dim=1, keepdim=True) # [B]: norm of each target sample
        P_norm = P_flat.norm(dim=1, keepdim=True) # [B]: norm of each prediction sample
        T_versor = T_flat / (T_norm + self.eps) # [B, D] 
        P_versor = P_flat / (P_norm + self.eps) # [B, D]
        cos_sim = (T_versor * P_versor).sum(dim=1)  # [B]: Cosine similarity, i.e, escalar product for each sample
        angle_diss = (1 - cos_sim)/2 # Errors for each sample
        loss = angle_diss**2 # Squared error for each sample
    
        return loss.mean() # Mean across samples
    
class Cosine_L1_Relative(nn.Module):
    def __init__(self, eps=1e-8):
        super(Cosine_L1_Relative, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")

        # Flattened versions for angle loss
        T_flat = target.flatten(start_dim=1).float()  # [B, D]
        P_flat = output.flatten(start_dim=1).float()  # [B, D]
        
        # Cosine similarity per sample
        T_norm = T_flat.norm(dim=1, keepdim=True) # [B]: norm of each target sample
        P_norm = P_flat.norm(dim=1, keepdim=True) # [B]: norm of each prediction sample
        T_versor = T_flat / (T_norm + self.eps) # [B, D] 
        P_versor = P_flat / (P_norm + self.eps) # [B, D]
        cos_sim = (T_versor * P_versor).sum(dim=1)  # [B]: Cosine similarity for each sample
        angle_diss = (1 - cos_sim)/2 # Errors for each sample
        #loss = angle_diss**2 # Squared error for each sample
        
        sample_mean = T_flat.abs().mean(dim=1, keepdim=True)  # [1, D]
        relative_error = (T_flat - P_flat).abs() / (sample_mean + 1e-8)  # [B, D]
        mag_diss = relative_error.norm(p=2, dim=1, keepdim=True)  # [B, 1]
        
        diss = mag_diss + mag_diss * angle_diss
        loss = diss**2 # Squared errors for all samples 
    
        return loss.mean() # Mean across samples
        

    
class MeanOutputError(nn.Module):
    def __init__(self):
        super(MeanOutputError, self).__init__()
        
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")
            
        mean_error = ( (output.mean() - target.mean())/target.mean() ).abs()
        return mean_error
        
class MeanPixelWiseRelativeError(nn.Module):
    """
    Computes the Mean Pixel-wise Relative Error (P-REL) Loss.
    
    This loss function is particularly useful for regression tasks where the
    magnitude of the target values varies significantly, such as depth
    estimation. It penalizes errors based on their proportion to the
    ground truth value, making it scale-invariant.
    
    Args:
        eps (float): A small value added to the denominator to prevent
                     division by zero when target pixel values are zero.
    """
    def __init__(self, eps=1e-8):
        super(MeanPixelWiseRelativeError, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        """
        Calculates the per-pixel relative error and averages it.

        Args:
            output (torch.Tensor): The predicted tensor, e.g., an image or depth map.
            target (torch.Tensor): The ground truth tensor of the same shape.

        Returns:
            torch.Tensor: The mean pixel-wise relative error loss.
        """
        # Ensure the output and target tensors have the same shape.
        if output.size() != target.size():
            raise ValueError(f"Shape mismatch: {output.size()} vs {target.size()}")    
        
        return (torch.abs(output - target) / (target.abs() + self.eps)).mean()
    
    
class MultiScaleLoss(nn.Module):
    # 'normalize_mode' can be : 
    #  - 'none' to return the raw sum, 
    #  - 'n_scales' to divide by the num. of scales and return the mean across scales
    #  - 'var' to divide by the variance of higher resolution's scale target
    def __init__(self, loss_fn, norm_mode='none'):
        """
        Args:
            loss_fn: any PyTorch loss function (e.g., nn.MSELoss(), nn.L1Loss())
        """
        super(MultiScaleLoss, self).__init__()
        self.loss_fn = loss_fn
        self.norm = norm_mode

    def forward(self, y_pred, y):
        """
        Args:
            y_pred (List[Tensor]): predictions at each scale
            y (List[Tensor]): ground truths at each scale
        Returns:
            loss: total multiscale loss
        """

        # Validate input types
        if not isinstance(y_pred, (list, tuple)):
            raise TypeError(f"Expected y_pred to be list or tuple, got {type(y_pred)}")
        if not isinstance(y, (list, tuple)):
            raise TypeError(f"Expected y to be list or tuple, got {type(y)}")
        if len(y_pred) != len(y):
            raise ValueError(f"Mismatch in number of scales: {len(y_pred)} predictions vs {len(y)} targets")
        
        total_loss = 0
        y_var = y[-1].var()

        for scale, (y_hats, y_trues) in enumerate(zip(y_pred, y)):
            if y_hats.shape != y_trues.shape:
                raise ValueError(f"Shape mismatch at scale {scale}: {y_hats.shape} vs {y_trues.shape}")
            
            # Get the scaled image loss, then include it to the total
            loss_scale = self.loss_fn(y_hats, y_trues) # n_voxels added by Gabriel C. Silveira
            total_loss += loss_scale # The total loss is the sum of each scale loss
            
        if self.norm    == 'none':      return total_loss
        elif self.norm  == 'n_scales':  return total_loss/len(y)
        elif self.norm  == 'var':       return total_loss/y_var
        else:
            raise ValueError(f"normalize_mode '{self.norm}' not implemented. Use one of 'none'(default), 'n_scales, 'var'.")
    
    

        
