import numpy as np
import Domain_Plotter as plt
from Utilities import array_handler as ah

def compare_input_target_output(loader, outputs):
    
    for i,(batch_inputs, batch_targets) in enumerate(loader):
        batch_output                       = outputs[0]  
        
        net_input_list  = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_inputs]
        net_target_list = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_targets]
        net_output_list = [tensor.squeeze(0).squeeze(0).detach().cpu().numpy() for tensor in batch_output]
        
        net_output_list = [ah.Filter_outlier(array) for array in net_output_list]
        net_output_list = [ah.Set_solids_to_value(output_array, input_array, 0) for output_array, input_array in zip(net_output_list, net_input_list) ]
        
        # Combine all arrays to find the global min and max for a shared color scale
        all_arrays = net_input_list + net_target_list + net_output_list
        global_min = min(arr.min() for arr in all_arrays)
        global_max = max(arr.max() for arr in all_arrays)
        
        # Plot inputs
        for j, array in enumerate(net_input_list):
            dimx, dimy, dimz = array.shape
            plt.Plot_Continuous_Domain_2D(
                values=array[:,:,0],
                filename=f"../NN_Results/Sample{i}/INPUT_Scale_{dimx}",
                remove_value=None,              # values to make transparent
                colormap="viridis",
                vmin=global_min,
                vmax=global_max,
                show_colorbar=True,
                special_colors={0: (1,1,1,1), 1: (0,0,0,1)}
            )
        # Plot targets
        for j, array in enumerate(net_target_list):
            dimx, dimy, dimz = array.shape
            plt.Plot_Continuous_Domain_2D(
                values=array[:,:,0],
                filename=f"../NN_Results/Sample{i}/TARGET_Scale_{dimx}",
                remove_value=None,              # values to make transparent
                colormap="viridis",
                vmin=global_min,
                vmax=global_max,
                show_colorbar=True,
                special_colors={0: (1,1,1,1)}
            )
        # Plot outputs
        for j, array in enumerate(net_output_list):
            dimx, dimy, dimz = array.shape
            plt.Plot_Continuous_Domain_2D(
                values=array[:,:,0],
                filename=f"../NN_Results/Sample{i}/OUTPUT_Scale_{dimx}",
                remove_value=None,              # values to make transparent
                colormap="viridis",
                vmin=global_min,
                vmax=global_max,
                show_colorbar=True,
                special_colors={0: (1,1,1,1)}
            )
        
        
def analyze_domain_error(loader, outputs):
    
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
        
        
        
        # For each sample's scale
        for j, (scale_net_input, scale_net_target, scale_net_output) in  enumerate(zip(net_input_list, net_target_list, net_output_list)):
            
            # Error
            error = np.abs(scale_net_output - scale_net_target)
            dimx, dimy, dimz = scale_net_output.shape
            
            ah.Set_solids_to_value(error, scale_net_input, -1)
            plt.Plot_Continuous_Domain_2D(
                values=error[:,:,0],
                title=r"Absolute Error: $e_{{i,j}} = |y_{{i,j}} - y_{{i,j}}^*|$",
                filename=f"../NN_Results/Sample{i}/Absolute_ERRO_Scale_{dimx}",
                remove_value=None,              # values to make transparent
                colormap="inferno",
                show_colorbar=True,
                special_colors={-1: (1,1,1,1)}
            )
            
            # Calculate relative error in log scale
            void_mask                   = scale_net_target != 0.0
            relative_error              = scale_net_target.copy()
            relative_error[void_mask]   = np.log2( (error[void_mask] / np.abs(scale_net_target[void_mask]))+1 )
            ah.Set_solids_to_value(relative_error, scale_net_input, -1)
            
            plt.Plot_Continuous_Domain_2D(
                values=relative_error[:,:,0],
                title="Relative Error in Void Space: $log_2(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$",
                filename=f"../NN_Results/Sample{i}/Absolute Log_REL_ERRO_Scale_{dimx}",
                remove_value=None,              # values to make transparent
                colormap="inferno",
                show_colorbar=True,
                special_colors={-1: (1,1,1,1)}
            )
            
            plt.plot_distributions(
                {"Relative Error in Void Space: $log_2(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$": relative_error[void_mask]},
                save_path=f"../NN_Results/Sample{i}/Distribution Log_REL_ERROR_Scale_{dimx}"
            )
            
            if dimx not in rel_error_all_samples:
                rel_error_all_samples[dimx] = []
            rel_error_all_samples[dimx].extend(relative_error[void_mask])
            
    for dimx, data in rel_error_all_samples.items():
        plt.plot_distributions(
            {"Relative Error in Void Space: $log_2(\\frac{|y_{{i,j}} - y_{{i,j}}^*|}{|y_{{i,j}}^*|}+1)$": np.array(data)},
            save_path=f"../NN_Results/Distribution Population_Log_REL_ERROR_Scale_{dimx}"
        )
            
def analyze_population(loader, outputs):

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
            scale_net_input  = scale_net_input.squeeze(0).squeeze(0).detach().cpu().numpy()
            scale_net_target = scale_net_target.squeeze(0).squeeze(0).detach().cpu().numpy()
            scale_net_output = scale_net_output.squeeze(0).squeeze(0).detach().cpu().numpy()            
            
            dimx, dimy, dimz = scale_net_input.shape
            scale = dimx
            
            # Calculate error
            void_mask   = scale_net_target!=0 
            error       = np.abs(scale_net_output - scale_net_target)
            rel_error   = np.log2( (error[void_mask] / np.abs(scale_net_target[void_mask]))+1 )
            
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
            net_input_inVoid[scale].extend(input_inVoid)
            net_target_inVoid[scale].extend(target_inVoid)
            net_output_inVoid[scale].extend(output_inVoid)
            net_error_inVoid[scale].extend(error_inVoid)
            net_relerror_inVoid[scale].extend(rel_error_inVoid)
      
            
    for scale in net_target_inVoid.keys():
        all_target          = np.array(net_target_inVoid[scale])
        all_input           = np.array(net_input_inVoid[scale])
        all_output          = np.array(net_output_inVoid[scale])
        rel_error_inVoid    = np.array(net_relerror_inVoid[scale])
        
        """
        plt.plot_distributions({"Target":all_target,"Prediction":all_output}, 
                               save_path = f"../NN_Results/Histogram_Pred_vs_True_Scale_{scale}",
                               title = ""
                               )
        """
        """
        plt.plot_joint_frequency(all_input, 
                                 all_target, 
                                 save_path=f"../NN_Results/JointFrequency_Dist_vs_PredVel_Scale_{scale}",
                                 title="", 
                                 xaxis="Absolute Distance", 
                                 yaxis="Absolute Target Velocity")
        """
        """
        plt.plot_joint_frequency(all_input,
                                 all_error,
                                 save_path=f"../NN_Results/JointFrequency_Dist_vs_AbsError_Scale_{scale}",
                                 title="",
                                 xaxis="Absolute Distance", 
                                 yaxis="Velocity Absolute Mismatch")
        """
        plt.plot_joint_frequency(all_input,
                                 rel_error_inVoid,
                                 save_path=f"../NN_Results/JointFrequency_Dist_vs_RelError_Scale_{scale}",
                                 title="",
                                 xaxis="Absolute Distance", 
                                 yaxis="Velocity Relative Error")
        
        plt.plot_joint_frequency(all_target,
                                 all_output,
                                 save_path=f"../NN_Results/JointFrequency_Target_vs_Output_{scale}",
                                 title="",
                                 xaxis="Target", 
                                 yaxis="Output")
        
        
    
    
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
            
            all_solid_is_zero       = ~np.any(scale_net_target[scale_net_input==0] != 0)
            all_void_has_velocity   = ~np.any(scale_net_target[scale_net_input!=0] != 0)
            
            if all_solid_is_zero and all_void_has_velocity: raise Exception("Error")
                
        