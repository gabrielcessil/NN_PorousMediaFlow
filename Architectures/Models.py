import torch
import torch.nn as nn
import numpy as np
from math import floor

from Architectures.FunctionalBlocks import (
    BASE_MODEL,
    ConvBlock,
    PoolingBlock,
    UpSampleBlock,
    ChannelConcat_Block
)


class Base_UNet(BASE_MODEL):
    def __init__(self, in_shape, out_shape, encoder, decoder, output_masks=None):
        super(Base_UNet, self).__init__(in_shape, out_shape, output_masks)
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.input_size = in_shape[1]

        self._validate_structure()

        self.encoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()

        self._build_encoder()
        self._build_decoder()

    def _validate_structure(self):
        if len(self.encoder_config) != len(self.decoder_config):
            raise ValueError("Encoder and decoder must have the same number of levels.")


    def _build_encoder(self):
        in_channels = self.in_channels
        size = self.input_size
        self.encoder_shapes = []

        for level in self.encoder_config:
            blocks = []
            for out_channels in level:
                block = ConvBlock(input_size=size, in_channels=in_channels, out_channels=out_channels, kernel_size=3)
                blocks.append(block)
                in_channels = out_channels
                size = block.output_size

            self.encoder_blocks.append(nn.Sequential(*blocks))
            self.encoder_shapes.append((in_channels, size))
            pool = PoolingBlock(input_size=size, in_channels=in_channels, kernel_size=2, stride=2)
            self.pooling_layers.append(pool)
            in_channels = pool.out_channels
            size = pool.output_size

    def _build_decoder(self):
        in_channels = self.encoder_shapes[-1][0]
        size = self.encoder_shapes[-1][1]

        for i, level in enumerate(self.decoder_config):
            up = UpSampleBlock(input_size=size, in_channels=in_channels, output_size=self.encoder_shapes[-(i+1)][1])
            self.upsample_layers.append(up)

            concat_channels = in_channels + self.encoder_shapes[-(i+1)][0]
            self.concat_layers.append(ChannelConcat_Block(input_size=up.output_size, in_channels=concat_channels))

            blocks = []
            for j, out_channels in enumerate(level):
                conv_in = concat_channels if j == 0 else level[j - 1]
                block = ConvBlock(input_size=up.output_size, in_channels=conv_in, out_channels=out_channels, kernel_size=3)
                blocks.append(block)

            self.decoder_blocks.append(nn.Sequential(*blocks))
            in_channels = level[-1]
            size = blocks[-1].output_size

        self.final_conv = ConvBlock(input_size=size, in_channels=in_channels, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for enc, pool in zip(self.encoder_blocks, self.pooling_layers):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        for up, concat, dec, skip in zip(self.upsample_layers, self.concat_layers, self.decoder_blocks, reversed(skips)):
            x = up(x)
            x = concat(x, skip)
            x = dec(x)

        x = self.final_conv(x)
        return x
    
# MS-NET
"""
The original code is present in :
    https://github.com/je-santos/ms_net
The original code to generate each individual conv model was modified from:
    https://github.com/tamarott/SinGAN
"""

class MS_Net(nn.Module):
    
    def __init__(
                 self, 
                 net_name     = 'test1', 
                 num_scales   =  4,
                 num_features =  1, 
                 num_filters  =  2, 
                 f_mult       =  4,  
                 summary      = False
                 ):
        
        super(MS_Net, self).__init__()
        
        self.net_name = net_name
        self.scales   = num_scales
        self.feats    = num_features
        
        self.models   = nn.ModuleList( 
                                MS_Net.get_SubModels( 
                                    num_scales,
                                    num_features,
                                    num_filters,
                                    f_mult ) 
                                )
        if summary:
            print(f'\n Here is a summary of your MS-Net ({net_name}): \n {self.models}')
        
    @staticmethod
    def get_SubModels(scales, features, filters, f_mult):
        
        """
        Returns an array with n-trainable models (ConvNets)
        """
        
        models   = []         # empty list to store the models
        nc_in    = features   # number of inputs on the first layer
        norm     = True       # use Norm
        last_act = None       # activation function
        
        # list of number filters in each model (scale)
        num_filters = [ filters*f_mult**scale for scale in range(scales) ][::-1]
       
        for it in range( scales ): # creates a model for each scale
            if it==1: nc_in+=1     # adds an additional input to the subsecuent models 
                                   # to convolve the domain + previous(upscaled) result 
            models.append( MS_Net.Scale_SubModel( nc_in    = nc_in,
                                      ncf      = num_filters[it],
                                      norm     = norm,
                                      last_act = last_act) )
        return models  
    
    def get_Masks(self, x, scales):
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
    
    
    def forward(self, x_list):
        # The coarsest network receives only the domain representation 
        # at the coarsest scale, while the subsequent ones receive two
        # the domain representation at the appropriate scale, 
        # and the prediction from the previous scale. 
        # As mentioned above, the input’s linear size is reduced by 
        # a factor of two between every scale.
        # x_list is the sample's input, a list of coarsened versions of an image
        
        masks   = self.get_Masks( (x_list[-1]>0).float(), self.scales)
        
        assert x_list[0].shape[1] == self.feats, \
        f'The number of features provided {x_list[0].shape[1]} \
            does not match with the input size {self.feats}'
            
        # Carry-out the first prediction (pass through the coarsest model)
        y = [ self.models[0]( x_list[0] ) ]
        for scale,[ model,x ] in enumerate(zip( self.models[1:],x_list[1:] )):
            y_up = self.scale_tensor( y[scale], scale_factor=2 )*masks[scale]
            y.append( model( torch.cat((x,y_up),dim=1) ) + y_up )
            
        return y
    
    def scale_tensor(self, x, scale_factor=1):
        
        if scale_factor<1:
            return nn.AvgPool3d(kernel_size = int(1/scale_factor))(x)
        elif scale_factor>1:
            for repeat in range (0, int(np.log2(scale_factor)) ):  # number of repeatsx2
                for ax in range(2,5): # (B,C,  H,W,D), repeat only the 3D axis, not batch and channel
                    x=x.repeat_interleave(repeats=2, axis=ax)
            return x
        elif scale_factor==1:
            return x
        
        else: raise ValueError(f"Scale factor not understood: {scale_factor}")
        
    class ConvBlock3D( nn.Sequential ):
        def __init__(self, in_channel, out_channel, ker_size, padd, stride, norm):
            super().__init__()
            self.add_module( 'conv',
                             nn.Conv3d( in_channel, 
                                        out_channel,
                                        kernel_size=ker_size,
                                        stride=stride,
                                        padding=padd ) ),
            if norm == True:
                self.add_module( 'i_norm', nn.InstanceNorm3d( out_channel ) ),
            self.add_module( 'CeLU', nn.CELU( inplace=False ) )

    class Scale_SubModel(nn.Module):
        def __init__(self, nc_in, ncf, norm, last_act):
            super().__init__()
            
            # default parameters
            nc_out     = 1   # number of output channels of the last layer
            ker_size   = 3   # kernel side-lenght
            padd_size  = 1   # padding size
            ncf_min    = ncf # min number of convolutional filters
            num_layers = 5   # number of conv layers
            
            # first block
            self.head = MS_Net.ConvBlock3D( 
                in_channel  = nc_in,
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
              
                convblock = MS_Net.ConvBlock3D( 
                    in_channel  = max(2*new_ncf,ncf_min),
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




# Danny D Ko
"""
The original code is present in :
    https://github.com/dko1217/DeepLearning-PorousMedia/tree/main
"""

class DannyKo_Net(nn.Module):
    class EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride, kernel_size, activation, momentum, dropout_rate):
            super().__init__()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
            self.norm = nn.BatchNorm3d(out_channels, momentum=momentum)
            self.act = nn.SELU(inplace=True) if activation == 'selu' else nn.ReLU(inplace=True)
            self.drop = nn.Dropout3d(dropout_rate)
    
        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)
            return x
    
    
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride, kernel_size, activation, momentum, dropout_rate):
            super().__init__()
            self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, output_padding=stride-1)
            self.norm = nn.BatchNorm3d(out_channels, momentum=momentum)
            self.act = nn.SELU(inplace=True) if activation == 'selu' else nn.ReLU(inplace=True)
            self.drop = nn.Dropout3d(dropout_rate)
    
        def forward(self, x):
            x = self.deconv(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)
            return x
    
    
    class UNetV1(nn.Module):
        def __init__(self, input_channels, filter_num=5, filter_size=3, activation='selu', momentum=0.99, dropout=0.2):
            super().__init__()
    
            self.encoder = nn.ModuleList()
            self.skip_connection_indices = []
            in_ch = input_channels
    
            # Encoder (8 layers total)
            for i in range(8):
                out_ch = int(filter_num * (2 ** floor(i / 2)))
                stride = 2 if i % 2 == 0 and i != 0 else 1
                self.encoder.append(self.EncoderBlock(in_ch, out_ch, stride, filter_size, activation, momentum, dropout))
                if i % 2 == 1:
                    self.skip_connection_indices.append(i)
                in_ch = out_ch
    
            # Decoder (6 layers total)
            self.decoder = nn.ModuleList()
            self.decoder_concat_indices = list(reversed(self.skip_connection_indices[:3]))  # only 3 skip connections used
    
            for i in reversed(range(6)):
                out_ch = int(filter_num * (2 ** floor(i / 2)))
                stride = 2 if i % 2 == 1 else 1
                self.decoder.append(self.DecoderBlock(in_ch, out_ch, stride, filter_size, activation, momentum, dropout))
                in_ch = out_ch * 2 if i % 2 == 1 else out_ch  # double for concatenated skip
    
            # Final convs
            self.conv_final1 = nn.Conv3d(in_ch, filter_num, kernel_size=filter_size, padding=filter_size//2)
            self.conv_final2 = nn.Conv3d(filter_num, 1, kernel_size=1)
    
        def forward(self, x):
            skips = []
            out = x
    
            # Encoder pass
            for i, block in enumerate(self.encoder):
                out = block(out)
                if i in self.skip_connection_indices:
                    skips.append(out)
    
            # Decoder pass
            for i, block in enumerate(self.decoder):
                out = block(out)
                if i % 2 == 1:
                    skip = skips[self.decoder_concat_indices[i // 2]]
                    out = torch.cat([out, skip], dim=1)
    
            out = self.conv_final1(out)
            out = self.conv_final2(out)
            return out
    
    
    class UNetV2(nn.Module):
        def __init__(self, input_channels, x_model, y_model, z_model,
                     filter_num=5, filter_size=3, activation='selu',
                     momentum=0.99, dropout=0.2, v_scale=1.0):
            super().__init__()
    
            self.x_model = x_model.eval()
            self.y_model = y_model.eval()
            self.z_model = z_model.eval()
            self.v_scale = v_scale
    
            self.encoder = nn.ModuleList()
            self.skip_connection_indices = []
            in_ch = 3  # Concatenated x, y, z outputs
    
            # Encoder
            for i in range(8):
                stride = 2 if i % 2 == 0 and i != 0 else 1
                out_ch = filter_num
                self.encoder.append(self.EncoderBlock(in_ch, out_ch, stride, filter_size, activation, momentum, dropout))
                if i % 2 == 1:
                    self.skip_connection_indices.append(i)
                in_ch = out_ch
    
            # Decoder
            self.decoder = nn.ModuleList()
            self.decoder_concat_indices = list(reversed(self.skip_connection_indices[:3]))
    
            for i in reversed(range(6)):
                stride = 2 if i % 2 == 1 else 1
                out_ch = filter_num
                self.decoder.append(self.DecoderBlock(in_ch, out_ch, stride, filter_size, activation, momentum, dropout))
                in_ch = out_ch * 2 if i % 2 == 1 else out_ch
    
            # Final convolution layers
            self.conv_final1 = nn.Conv3d(in_ch, filter_num, kernel_size=filter_size, padding=filter_size // 2)
            self.conv_final2 = nn.Conv3d(filter_num, 3, kernel_size=1)
    
        def forward(self, x):
            with torch.no_grad():
                x_input = 0.5 * self.v_scale * self.x_model(x)
                y_input = 0.5 * self.v_scale * self.y_model(x)
                z_input = 1.0 * self.v_scale * self.z_model(x)
    
            path = torch.cat([x_input, y_input, z_input], dim=1)
    
            skips = []
            out = path
    
            # Encoder
            for i, block in enumerate(self.encoder):
                out = block(out)
                if i in self.skip_connection_indices:
                    skips.append(out)
    
            # Decoder
            for i, block in enumerate(self.decoder):
                out = block(out)
                if i % 2 == 1:
                    skip = skips[self.decoder_concat_indices[i // 2]]
                    out = torch.cat([out, skip], dim=1)
    
            out = self.conv_final1(out)
            out = self.conv_final2(out)
            return out
    

