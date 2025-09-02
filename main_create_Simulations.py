from Utilities import LBPM_runner 
import json

    
#######################################################
#************ LOAD CONFIGS                 ***********#
#######################################################

# Load configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)

simulations_main_folder = config_loaded["LBPM_IO_folder"]
lbm_file_path = config_loaded["LBPM_installation_folder"]




#######################################################
#************ COMPUTATIONS                 ***********#
####################################################### 

# Any folder inside simulations_main_folder must be name a example and the example's files must be follow the folder name 
examples_names = LBPM_runner.get_folder_names(simulations_main_folder)


# Run every folder example
for simulation_name in examples_names:
    print("\n\nRunning simulation: ", simulation_name)
    LBPM_runner.Run_Example(simulations_main_folder,
                             simulation_name,
                             lbm_file_path)