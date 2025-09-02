from Architectures import Inception_v3

"""
enc_level1_pool = Inception_v3.PoolingBlock(
    input_size=500,
    output_size=250,
    padding=1,    
    in_channels=4,
    kernel_size=3,
    stride=2
    )
print("Output: ",enc_level1_pool.output_size)


enc_level1_pool = Inception_v3.PoolingBlock(
    input_size=500,
    padding=1,    
    in_channels=4,
    kernel_size=3,
    stride=2
    )
print("Output automatico: ",enc_level1_pool.output_size)



enc_level1_pool = Inception_v3.PoolingBlock(
    input_size=500,
    output_size=250,
    in_channels=4,
    kernel_size=3,
    stride=2
    )
print("Output para Padding automatico: ",enc_level1_pool.output_size)


enc_level1_pool = Inception_v3.ConvTransposeBlock(
    input_size=250,
    output_size=500,
    in_channels=4,
    out_channels=4,

    output_padding=1,
    dilation=1,
    padding=1,    
    kernel_size=3,
    stride=2
    )
print("Output: ",enc_level1_pool.output_size)



enc_level1_pool = Inception_v3.ConvTransposeBlock(
    input_size=250,
    output_size=500,
    in_channels=4,
    out_channels=4,
    
    dilation=1,
    padding=1,    
    kernel_size=3,
    stride=2
    )
print("Output para Padding automatico: ",enc_level1_pool.output_size)
"""
import torch
random_input = torch.rand(1,3,250,250)
random_estimation = torch.rand(1,5,250,250)

"""
# Testing forward 
block = Inception_v3.BLOCK_2(in_shape=(3,250,250),
                              out_shape=(5,250,250),
                              output_masks=None,
                              enc_decay=2,
                              add_channels=6,
                              estimative_signal=False)
block.forward(random_input)


# Testing 
block = Inception_v3.BLOCK_2(in_shape=(3,250,250),
                              out_shape=(5,250,250),
                              output_masks=None,
                              enc_decay=2,
                              add_channels=6,
                              estimative_signal=True)
block.forward(random_input, estimation=random_estimation)

"""
"""
block = Inception_v3.MULTI_BLOCK_MODEL_2(
            N_blocks=3,
            in_shape=(3,250,250),
            out_shape=(5,250,250),
            output_masks=None,
            enc_decay=2,
            add_channels=6,
            estimative_signal=False)
block.forward(random_input)
"""
block = Inception_v3.MULTI_BLOCK_MODEL_2(
            N_blocks=3,
            in_shape=(3,250,250),
            out_shape=(5,250,250),
            output_masks=None,
            enc_decay=2,
            add_channels=6,
            estimative_signal=True)
block.forward(random_input, random_estimation)
metadata = block.metadata