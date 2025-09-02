import numpy as np
import torch
import torch.nn as nn



def scale_tensor(x, scale_factor=1):
    
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
        

def get_masks(x, scales):
    """
    x: euclidean distance 3D array at the finest scale
    Returns array with masks
    
    Notes:
        for n scales we need n masks (the last one is binary)
    """    
    masks    = [None]*(scales)
    pooled   = [None]*(scales)
    
    pooled[0] = (x>0).float() # 0s at the solids, 1s at the empty space
    masks[0]  = pooled[0].squeeze(0)
    
    for scale in range(1,scales):
        pooled[scale] = nn.AvgPool3d(kernel_size = 2)(pooled[scale-1])
        denom = pooled[scale].clone()   # calculate the denominator for the mask
        denom[denom==0] = 1e8  # regularize to avoid divide by zero
        for ax in range(2,5):   # repeat along 3 axis
            denom=denom.repeat_interleave( repeats=2, axis=ax ) # Upscale
        # Calculate the mask as Mask = Image / Upscale( Downscale(Img) )
        masks[ scale ] = torch.div( pooled[scale-1], denom ).squeeze(0) 
    return masks[::-1] # returns a list with masks. smallest size first
        

