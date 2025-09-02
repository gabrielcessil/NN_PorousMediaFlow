import numpy as np
import torch
import torch.nn as nn
from pore_utils import get_coarsened_list
from network_tools import get_masks
from network_tools import scale_tensor

"""
The MS-Net 
A Computationally Efficient Multiscale Neural Network 


The code to generate each individual conv model was modified from:
https://github.com/tamarott/SinGAN

"""


def get_trainable_models(scales, features, filters, f_mult):
    
    """
    Returns an array with n-trainable models (ConvNets)
    """
    
    models   = []         # empty list to store the models
    nc_in    = features   # number of inputs on the first layer
    norm     = True       # use Norm
    last_act = None       # activation function
    
    # list of number filters in each model (scale)
    num_filters = [ filters*f_mult**scale for scale in range(scales) ][::-1]
    print(f'Filters per model: {num_filters}')
   
    for it in range( scales ): # creates a model for each scale
        if it==1: nc_in+=1     # adds an additional input to the subsecuent models 
                               # to convolve the domain + previous(upscaled) result 
        models.append( get_model( nc_in    = nc_in,
                                  ncf      = num_filters[it],
                                  norm     = norm,
                                  last_act = last_act) )
    return models  


class get_model(nn.Module):
    def __init__(self, nc_in, ncf, norm, last_act):
        super(get_model, self).__init__()
        
        # default parameters
        nc_out     = 1   # number of output channels of the last layer
        ker_size   = 3   # kernel side-lenght
        padd_size  = 1   # padding size
        ncf_min    = ncf # min number of convolutional filters
        num_layers = 5   # number of conv layers
        
        # first block
        self.head = ConvBlock3D( in_channel  = nc_in,
                                 out_channel = ncf,
                                 ker_size    = ker_size,
                                 padd        = padd_size,
                                 stride      = 1,
                                 norm        = norm )
        
        # body of the model
        self.body = nn.Sequential()
        for i in range( num_layers-1 ):
            new_ncf = int( ncf/2**(i+1) )
            if i==num_layers-2: norm=False  # no norm in the penultimate block
          
            convblock = ConvBlock3D( in_channel  = max(2*new_ncf,ncf_min),
                                     out_channel = max(new_ncf,ncf_min),
                                     ker_size    = ker_size,
                                     padd        = padd_size,
                                     stride      = 1,
                                     norm        = norm)
            
            self.body.add_module( f'block{i+1}', convblock )
        
        if last_act == 'CELU':
            self.tail = nn.Sequential(
                                    nn.Conv3d( max(new_ncf,ncf_min), nc_out,
                                               kernel_size=1,stride=1, padding=0),
                                    nn.CELU()
                                 )
        else:
            self.tail = nn.Sequential(
                            nn.Conv3d( max(new_ncf,ncf_min), nc_out, kernel_size=1,
                                       stride=1, padding=0)) # no pad needed since 1x1x1
            
    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x




class ConvBlock3D( nn.Sequential ):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm):
        super(ConvBlock3D,self).__init__()
        self.add_module( 'conv',
                         nn.Conv3d( in_channel, 
                                    out_channel,
                                    kernel_size=ker_size,
                                    stride=stride,
                                    padding=padd ) ),
        if norm == True:
            self.add_module( 'i_norm', nn.InstanceNorm3d( out_channel ) ),
        self.add_module( 'CeLU', nn.CELU( inplace=False ) )
        
        

class MS_Net(nn.Module):
    
    def __init__(
                 self, 
                 net_name     = 'test1', 
                 num_scales   =  4,
                 num_features =  1, 
                 num_filters  =  2, 
                 f_mult       =  4,  
                 device       =  'cpu',
                 
                 summary      = False
                 ):
        
        super(MS_Net, self).__init__()
        
        self.net_name = net_name
        self.scales   = num_scales
        self.feats    = num_features
        self.device   = device
        
        self.models   = nn.ModuleList( 
                                get_trainable_models( num_scales,
                                                      num_features,
                                                      num_filters,
                                                      f_mult ) )
        if summary:
            print(f'\n Here is a summary of your MS-Net ({net_name}): \n {self.models}')
        
        
        
    # x_list is the sample's input, a list of coarsened versions of an image
    def forward(self, x_list):
        
        # The coarsest network receives only the domain representation 
        # at the coarsest scale, while the subsequent ones receive two
        # the domain representation at the appropriate scale, 
        # and the prediction from the previous scale. 
        # As mentioned above, the input’s linear size is reduced by 
        # a factor of two between every scale.
        
        masks   = get_masks( (x_list[-1]>0).float(), self.scales)
        
        assert x_list[0].shape[1] == self.feats, \
        f'The number of features provided {x_list[0].shape[1]} \
            does not match with the input size {self.feats}'
            
        # Carry-out the first prediction (pass through the coarsest model)
        y = [ self.models[0]( x_list[0] ) ]
        for scale,[ model,x ] in enumerate(zip( self.models[1:],x_list[1:] )):
            y_up = scale_tensor( y[scale], scale_factor=2 )*masks[scale]
            y.append( model( torch.cat((x,y_up),dim=1) ) + y_up )
            
        return y
