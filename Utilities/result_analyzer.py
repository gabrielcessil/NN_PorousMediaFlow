import numpy as np
import Domain_Plotter as pl
from Utilities import array_handler as ah
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def analyze_input_target_output_domain(loader, outputs, path):
    
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output                       = outputs[i]  
        
        net_input_list  = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_inputs]
        net_target_list = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_targets]
        net_output_list = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_output]
        
        net_output_list = [ah.Filter_outlier(array) for array in net_output_list]
        net_output_list = [ah.Set_solids_to_value(output_array, input_array, 0) for output_array, input_array in zip(net_output_list, net_input_list) ]
        
        # Combine all arrays to find the global min and max for a shared color scale
        all_arrays = net_target_list + net_output_list
        global_min = min(arr.min() for arr in all_arrays)
        global_max = max(arr.max() for arr in all_arrays)
        
        # Plot inputs
        
        for j, (arr_input, arr_target, arr_output) in enumerate(zip(net_input_list, net_target_list, net_output_list)):
            dimx, dimy, dimz = arr_input.shape
            
            
            pl.Plot_Domain((arr_input!=0).astype(int), path+f"Sample{i}/{dimx}_input", remove_value=[1])
            pl.Plot_Domain((arr_target!=0).astype(int), path+f"Sample{i}/{dimx}_target", remove_value=[1])
            pl.Plot_Domain((arr_output!=0).astype(int), path+f"Sample{i}/{dimx}_output", remove_value=[1])
            
            pl.Plot_Continuous_Domain_2D(
                values=arr_input[0,:,:],
                filename=path+f"Sample{i}/{dimx}_INPUT_Scale",
                colormap="viridis",
                vmin=np.min(arr_input[0,:,:]),
                vmax=np.max(arr_input[0,:,:]),
                show_colorbar=True,
                special_colors={0: (1,1,1,1), 1: (0,0,0,1)}
            )
            pl.Plot_Continuous_Domain_2D(
                values=arr_target[0,:,:],
                filename=path+f"Sample{i}/{dimx}_TARGET_Scale",
                colormap="viridis",
                vmin=global_min,
                vmax=global_max,
                show_colorbar=True,
                special_colors={0: (1,1,1,1)}
            )
            pl.Plot_Continuous_Domain_2D(
                values=arr_output[0,:,:],
                filename=path+f"Sample{i}/{dimx}_OUTPUT_Scale",
                remove_value=None,              # values to make transparent
                colormap="viridis",
                vmin=global_min,
                vmax=global_max,
                show_colorbar=True,
                special_colors={0: (1,1,1,1)}
            )
        
        
        output_slice                = arr_output[dimz//2,:,:].copy()
        target_slice                = arr_target[dimz//2,:,:].copy()
        plot_array                  = output_slice.copy()
        fluid_mask                  = target_slice != 0
        plot_array[fluid_mask]     = 100* np.abs(  (output_slice[fluid_mask] - target_slice[fluid_mask]) / (target_slice[fluid_mask]) )
        plot_array[~fluid_mask]    = -1
        pl.Plot_Continuous_Domain_2D(
            values=plot_array,
            filename=path+f"Sample{i}/{dimx}_Percentual Error",
            colormap="jet",
            show_colorbar=True,
            vmax=100,
            vmin=0,
            special_colors={-1: (1,1,1,1)}
        )
        
        pl.plot_line_in_domain(arr_target, arr_output, save_path=path+f"Sample{i}/{dimx}_outputLine", along="z")
        print(path+f"Sample{i}/{dimx}_outputLine")
            
            
            

        
def analyze_domain_error(loader, outputs, path):
    
    # Performance of each sample
    rel_error_all_samples = {}
    # For each sample
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output = outputs[i]
        
        net_input_list  = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_inputs]
        net_target_list = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_targets]
        net_output_list = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_output]
        
        net_output_list = [ah.Filter_outlier(array) for array in net_output_list]
        net_output_list = [ah.Set_solids_to_value(output_array, input_array, 0) for output_array, input_array in zip(net_output_list, net_input_list) ]
        
        
        print(f" Sample {i}")
        # For each sample's scale
        for j, (scale_net_input, scale_net_target, scale_net_output) in  enumerate(zip(net_input_list, net_target_list, net_output_list)):
            
            # Error
            error               = np.abs(scale_net_output - scale_net_target)
            dimx, dimy, dimz    = scale_net_output.shape
            error_norm          = error / np.std(scale_net_target)
            
            
            
            
            
            plotted_error = ah.Set_solids_to_value(error_norm, scale_net_input, -1)
            pl.Plot_Continuous_Domain_2D(
                values=plotted_error[0,:,:],
                title=r"Absolute Error: $e_{{i,j}} = |y_{{i,j}} - y_{{i,j}}^*| / std(y_{{i,j}}^*)$",
                filename=path+f"Sample{i}/Absolute_ERRO_Scale_{dimx}",
                colormap="inferno",
                show_colorbar=True,
                special_colors={-1: (1,1,1,1)},
                vmin=0
            )
            
            # Calculate relative error in log scale
            void_mask                   = scale_net_target != 0.0
            relative_error              = scale_net_target.copy()
            relative_error[void_mask]   = np.log2( (error[void_mask] / np.abs(scale_net_target[void_mask]))+1 )
            
            print("error: min:              ",  np.min(error),", max: ", np.max(error))
            print("Normalized error: min:   ",  np.min(error_norm),", max: ", np.max(error_norm))
            print("Relative error: max:     ",  np.max(  np.abs(scale_net_target[void_mask])  ),", min: ",np.min(  np.abs(scale_net_target[void_mask])  ))
            print()
            
            ah.Set_solids_to_value(relative_error, scale_net_target, -1)
            pl.Plot_Continuous_Domain_2D(
                values=relative_error[0,:,:],
                title="Relative Error in Void Space: $log_2(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$",
                filename=path+f"Sample{i}/Absolute Log_REL_ERRO_Scale_{dimx}",
                colormap="inferno",
                show_colorbar=True,
                special_colors={-1: (1,1,1,1)}
            )
            
            pl.plot_distributions(
                {"Relative Error in Void Space: $log_2(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$": relative_error[void_mask]},
                save_path=path+f"Sample{i}/Distribution Log_REL_ERROR_Scale_{dimx}"
            )
            
            if dimx not in rel_error_all_samples:
                rel_error_all_samples[dimx] = []
            rel_error_all_samples[dimx].extend(relative_error[void_mask])
            
        
            
    for dimx, data in rel_error_all_samples.items():
        pl.plot_distributions(
            {"Relative Error in Void Space: $log_2(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$": np.array(data)},
            save_path=path+f"Distribution Population_Log_REL_ERROR_Scale_{dimx}"
        )
        
            
def analyze_population_distributions(loader, outputs, path):

    # Performance of all samples (per scale)        
    net_target_inVoid = {} 
    net_output_inVoid = {}
    net_input_inVoid  = {}
    net_error_inVoid = {}
    net_relerror_inVoid = {}
    
    # For each sample
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output = outputs[i]
        # For each sample's scale
        for scale_net_input, scale_net_target, scale_net_output in  zip(batch_inputs, batch_targets, batch_output):
 
            # Convert tensors to array
            scale_net_input     = scale_net_input.squeeze(0).squeeze(0).detach().cpu().numpy()
            scale_net_target    = scale_net_target.squeeze(0).squeeze(0).detach().cpu().numpy()
            scale_net_output    = scale_net_output.squeeze(0).squeeze(0).detach().cpu().numpy()            
            
            dimx, dimy, dimz    = scale_net_input.shape
            scale               = dimx
            
            # Calculate error
            void_mask           = scale_net_target!=0 
            error               = np.abs(scale_net_output - scale_net_target)
            rel_error           = error[void_mask] / np.abs(scale_net_target[void_mask])
            
            target_inVoid       = scale_net_target[void_mask].tolist()
            input_inVoid        = scale_net_input[void_mask].tolist()
            output_inVoid       = scale_net_output[void_mask].tolist()
            error_inVoid        = error[void_mask].tolist()
            rel_error_inVoid    = rel_error.tolist()
            
            # Extend the list of values to compute all the samples at same plot
            if scale not in net_target_inVoid:
                net_target_inVoid[scale]    = []
                net_input_inVoid[scale]     = []
                net_output_inVoid[scale]    = []
                net_error_inVoid[scale]     = []
                net_relerror_inVoid[scale]  = []
            net_input_inVoid[scale].extend(    input_inVoid/ np.std(input_inVoid))
            net_target_inVoid[scale].extend(   target_inVoid/ np.std(target_inVoid))
            net_output_inVoid[scale].extend(   output_inVoid/ np.std(output_inVoid))
            net_error_inVoid[scale].extend(    error_inVoid/ np.std(error_inVoid))
            net_relerror_inVoid[scale].extend( rel_error_inVoid/ np.std(rel_error_inVoid))
      
            
    for scale in net_target_inVoid.keys():
        all_target          = np.array(net_target_inVoid[scale])
        all_input           = np.array(net_input_inVoid[scale])
        all_output          = np.array(net_output_inVoid[scale])
        rel_error_inVoid    = np.array(net_relerror_inVoid[scale])
        
        
        npoints=5000
        pl.plot_scatter_sampled(
            all_target,all_output,
            npoints=npoints,
            xlabel="Target", ylabel="Output",
            title=f'2D Scatter Plot: Target vs. Prediction ({npoints} Sampled Points)',
            save_path=path+f"TargetVsOuput_{scale}")
        
        pl.plot_distributions(
            {"Targets": all_target,
             "Outputs": all_output,
             },
            save_path=path+f"TargetsVsOutputs_Histogram_{scale}",
            normalize=True
        )

        
        
    
    
def sanity_check(loader, outputs):
    # For each sample
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output = outputs[i]
        # For each sample's scale
        for scale, (scale_net_input, scale_net_target, scale_net_output) in  enumerate(zip(batch_inputs, batch_targets, batch_output)):
            # Convert tensors to array
            scale_net_input  = scale_net_input.squeeze(0).squeeze(0).detach().cpu().numpy()
            scale_net_target = scale_net_target.squeeze(0).squeeze(0).detach().cpu().numpy()
            scale_net_output = scale_net_output.squeeze(0).squeeze(0).detach().cpu().numpy() 
            
            solids_True             = scale_net_input==0
            voids_True              = scale_net_input!=0
            any_solid_w_velocity    = not np.any(scale_net_target[solids_True] != 0) # If any velocity is non zero in solid, 
            any_void_wo_velocity    = not np.any(scale_net_target[voids_True] == 0)  # If there is any void without velocity
            
            if any_solid_w_velocity: raise Exception("Error: some solid cell (from Input) has velocity in Target")
            if any_void_wo_velocity: raise Exception("Error: some void cell (from Input) has no velocity in Target")
                
        