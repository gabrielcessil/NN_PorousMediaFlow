import torch
import h5py
from scipy.io import loadmat
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import Domain_Plotter as plt
import pyvista as pv
import numpy.typing as npt
import re
import copy
###############################################################################
#************************************ DATASET INFOS **************************#
# Dataset:
#   https://digitalporousmedia.org/published-datasets/tapis/projects/drp.project.published/drp.project.published.DRP-372
#   Format: .mat
#   Convention in data: Solid=1, Void=0; Converted to Solid=0, Void=1 (LBPM Convention)
###############################################################################



def save_velocity_field_vti(dist, ux, uy, uz, filename="velocity", spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Save a 3D velocity field (ux, uy, uz) as a ParaView-readable .vti file.

    Parameters
    ----------
    ux, uy, uz : np.ndarray
        3D arrays of the velocity components. Shape should be (Nz, Ny, Nx).
    filename : str
        target file name (default: "velocity.vti").
    spacing : tuple of float
        Grid spacing along (dx, dy, dz). Default is (1, 1, 1).
    origin : tuple of float
        Origin of the grid in physical coordinates. Default is (0, 0, 0).
    """

    # 1. Ensure the arrays have the same shape
    if not (ux.shape == uy.shape == uz.shape):
        raise ValueError("Input velocity component arrays must have the same shape.")

    # 2. Get the dimensions and total number of points from the arrays
    Nz, Ny, Nx = ux.shape
    num_points = Nx * Ny * Nz

    # 3. Create a 3D numpy array of vectors. The shape should be (N, 3) where N is the total number of points.
    # PyVista automatically handles the flattening and ordering, but it's good practice to do it explicitly.
    # The arrays are flattened in C-order (row-major), which is the default for NumPy and matches VTK.
    velocity_vector_field = np.stack([ux.ravel(), uy.ravel(), uz.ravel()], axis=1)

    # 4. Create the uniform grid
    grid = pv.ImageData(dimensions=(Nx, Ny, Nz), spacing=spacing, origin=origin)
    
    # 5. Attach the velocity data
    # The 'velocity' field is a vector field and must have 3 components per point.
    grid.point_data["Velocity"]             = velocity_vector_field
    grid.point_data["Velocity_x"]           = ux.ravel()
    grid.point_data["Velocity_y"]           = uy.ravel()
    grid.point_data["Velocity_z"]           = uz.ravel()
    grid.point_data["SignDist"]             = dist.ravel()
    grid.point_data["Velocity_magnitude"]   = np.linalg.norm(velocity_vector_field, axis=1)
    
    # 5. Attach the solid data, ensuring it's a scalar array
    # The original 'uz_c == 0' logic works, but let's stick to the C-order convention.
    solid_data = (dist > 0).astype(np.uint8)
    grid.point_data["solid"] = solid_data.ravel(order='C')
    
    # 6. Save the grid to a single .vti file
    grid.save(f"{filename}.vti")
    print(f"Successfully saved {filename}.vti with velocity and solid data.")

            
from torch.utils.data import Dataset
class Lazy_RockDataset(Dataset):
    def __init__(self, filespath):
        
        all_paths, all_paths_480 = self.get_PathPairs(filespath)

        self.conv_factors = {
            1: 1.619499463,
            2: 0.790754568,
            5: 0.29330618,
            10: 0.128986101,
            20: 0.051011612,
            }
        self.paths = all_paths
    
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        rock_filename, simu_filename = self.paths[idx]
        pattern = re.compile(r"P_(\d+)_?MPa", flags=re.IGNORECASE)

        match = pattern.search(simu_filename)
        if match:
            value = int(match.group(1))  # pega só o número
        
        try:
            rock_dist_trans, velocity_target = self.get_SamplePair(rock_filename, simu_filename, value)
            
            # Add batch dimension
            example_input_tensor    = torch.tensor(rock_dist_trans.astype(np.float32)).unsqueeze(0)
            example_target_tensor   = torch.tensor(velocity_target.astype(np.float32)).unsqueeze(0) 
            return example_input_tensor, example_target_tensor

        except Exception as e:
            print(f"Error loading Lazy_RockDataset index {idx}, i.e, path {rock_filename}: {e}")
            return None, None  # You may need to handle this in your DataLoader (e.g., by skipping None)

    def sanity_check(self,rock, target, solid_default_value=0):
        # SANITY CHECK: NO VELOCITY IN ROCK SOLID
        solid_mask = rock==solid_default_value
        return np.mean(abs(target[solid_mask]))==0
        
    def get_SamplePair(self, rock_filename, simu_filename, pressure):
         
        with h5py.File(rock_filename, 'r') as f: rock_bin = f["bin"][()] # shape (z,y,z)
            
        # Rotation needed in the dataset provided
        if not np.isin(rock_bin, [0, 1]).all(): raise Exception("Rock file must be binary int.")
        rock_bin          = 1-rock_bin                                                          # Revert convertion to 0-solid, 1-fluid
        rock_bin          = rock_bin.transpose(2, 1, 0)                                         
        rock_dist_trans     = distance_transform_edt(rock_bin == 1.0)                             
        
        simu_data           = loadmat(simu_filename)                                                # Field alligned with i (z,y,x)

        # (ORIGINAL FORM)
        example_target_ux   = simu_data["ux"][()] *self.conv_factors[pressure] 
        example_target_uy   = simu_data["uy"][()] *self.conv_factors[pressure]
        example_target_uz   = simu_data["uz"][()] *self.conv_factors[pressure]
        

        #example_target_ux   = simu_data["ux"][()] 
        #example_target_uy   = simu_data["uy"][()] 
        #example_target_uz   = simu_data["uz"][()] 
        
        
        if not self.sanity_check(rock_bin, example_target_uz, solid_default_value=0.0): 
            raise Exception(f"Sanity check: Target and Rock domains do not match in solid cells: {rock_filename} and {simu_filename}")
    
        # NORMALIZATION (ORIGINAL FORM)
        rock_dist_trans     = rock_dist_trans
        example_target_ux   = example_target_ux/1e-9
        example_target_uy   = example_target_uy/1e-9
        example_target_uz   = example_target_uz/1e-9
        
        #rock_dist_trans     = rock_dist_trans/100
        #example_target_ux   = example_target_ux/1e-6
        #example_target_uy   = example_target_uy/1e-6
        #example_target_uz   = example_target_uz/1e-6
        
        # SWAP X AND Z AXIS of each scalar field
        rock_bin            = np.rot90(rock_bin, axes=(0, 2))  
        rock_bin            = np.flip(rock_bin, axis=0)
        rock_dist_trans     = np.rot90(rock_dist_trans, axes=(0, 2))  
        rock_dist_trans     = np.flip(rock_dist_trans, axis=0)
        example_target_ux   = np.rot90(example_target_ux, axes=(0, 2))
        example_target_ux   = np.flip(example_target_ux, axis=0)
        example_target_uy   = np.rot90(example_target_uy, axes=(0, 2))
        example_target_uy   = np.flip(example_target_uy, axis=0)
        example_target_uz   = np.rot90(example_target_uz, axes=(0, 2))
        example_target_uz   = np.flip(example_target_uz, axis=0)
        
        #plt.Plot_Domain((rock_dist_trans!=0).astype(int), "input_0", remove_value=[1])
        #plt.Plot_Domain((example_target_uz!=0).astype(int),"target_0", remove_value=[1])
        
        # Visualize
        vti_filename = rock_filename.removesuffix('.mat')
        save_velocity_field_vti(                                                
            rock_dist_trans,
            example_target_ux,    
            example_target_uy,  
            example_target_uz,
            filename=vti_filename+f"_{pressure}", 
            spacing=(1.0, 1.0, 1.0), 
            origin=(0.0, 0.0, 0.0))
       
        # Make raw
        raw_filename    = rock_filename.removesuffix('.mat')
        data            = rock_bin.astype(np.uint8)
        with open(raw_filename+".raw", 'wb') as f:
            # Write the raw binary data of the array to the file
            data.tofile(f)
        print(f"Successfully saved raws with shape {data.shape} data as: {raw_filename}")

        # Make flipped and walled raw
        # For periodic simulation: reflect the data along the z-axis if requested
        raw_filename    = raw_filename+"_flipped"
        flipped         = np.flip(data, axis=0)
        data            = np.concatenate([data, flipped], axis=0)
        # Make walls in bounder YZ and XZ planes
        data[:, :, 0]   = 0
        data[:, :, -1]  = 0
        data[:, 0, :]   = 0
        data[:, -1, :]  = 0
        # Open the file in binary write mode
        with open(raw_filename+".raw", 'wb') as f:
            # Write the raw binary data of the array to the file
            data.tofile(f)
            
        print(f"Successfully saved raws with shape {data.shape} data as: {raw_filename}")
        
        return rock_dist_trans, example_target_uz
    
    
    # Some downloads ended uo in the folder, others in a folder with same name
    def resolve_mat_path(self, base_path):
        # Try direct file first
        try:
            
            if os.path.isfile(base_path):
                return base_path
            
            # Try if it's a directory containing the same-named .mat file
            elif os.path.isdir(base_path):
                candidate = os.path.join(base_path, os.path.basename(base_path))
                if os.path.isfile(candidate):
                    return candidate
                else:
                    raise Exception(f"Could not resolve .mat inside base_path folder: {candidate} is not a file.")
            else:
                raise Exception(f"Could not resolve .mat file for base_path: {base_path} is neither a file or folder.")
                
        except Exception as e:
            raise Exception(f"Could not resolve .mat file in {base_path}")
        
    def get_PathPairs(self,RELATIVE_ROOT):
        """
        # Modified FinneyPacks
        f"/{RELATIVE_ROOT}/374_01_00/", 
        f"/{RELATIVE_ROOT}/374_01_01/",
        f"/{RELATIVE_ROOT}/374_01_02/",
        f"/{RELATIVE_ROOT}/374_01_03/",
        f"/{RELATIVE_ROOT}/374_01_04/",
        f"/{RELATIVE_ROOT}/374_01_05/",
        f"/{RELATIVE_ROOT}/374_01_06/",
        f"/{RELATIVE_ROOT}/374_01_07/",
        f"/{RELATIVE_ROOT}/374_01_08/",
        
        # Fractures - Vuggy FinneyPacks
        f"/{RELATIVE_ROOT}/374_02_00/",
        f"/{RELATIVE_ROOT}/374_02_01/",
        f"/{RELATIVE_ROOT}/374_02_010/",
        f"/{RELATIVE_ROOT}/374_02_011/",
        f"/{RELATIVE_ROOT}/374_02_012/",
        f"/{RELATIVE_ROOT}/374_02_013/",
        f"/{RELATIVE_ROOT}/374_02_014/",
        f"/{RELATIVE_ROOT}/374_02_02/",
        f"/{RELATIVE_ROOT}/374_02_020/",
        f"/{RELATIVE_ROOT}/374_02_021/",
        f"/{RELATIVE_ROOT}/374_02_022/",
        f"/{RELATIVE_ROOT}/374_02_023/",
        f"/{RELATIVE_ROOT}/374_02_024/",
        f"/{RELATIVE_ROOT}/374_02_03/",
        f"/{RELATIVE_ROOT}/374_02_030/",
        f"/{RELATIVE_ROOT}/374_02_031/", 
        f"/{RELATIVE_ROOT}/374_02_032/",
        f"/{RELATIVE_ROOT}/374_02_033/",
        f"/{RELATIVE_ROOT}/374_02_034/",
        f"/{RELATIVE_ROOT}/374_02_04/",
        
        # Process-based packs - Single Fractures with Variable Aperture and Roughness
        f"/{RELATIVE_ROOT}/374_03_01/",
        f"/{RELATIVE_ROOT}/374_03_010/",
        f"/{RELATIVE_ROOT}/374_03_011/",
        f"/{RELATIVE_ROOT}/374_03_012/",
        f"/{RELATIVE_ROOT}/374_03_013/",
        f"/{RELATIVE_ROOT}/374_03_014/",
        f"/{RELATIVE_ROOT}/374_03_015/",
        f"/{RELATIVE_ROOT}/374_03_016/",
        f"/{RELATIVE_ROOT}/374_03_017/",
        f"/{RELATIVE_ROOT}/374_03_018/",
        f"/{RELATIVE_ROOT}/374_03_02/",
        f"/{RELATIVE_ROOT}/374_03_03/",
        f"/{RELATIVE_ROOT}/374_03_04/",
        f"/{RELATIVE_ROOT}/374_03_05/",
        f"/{RELATIVE_ROOT}/374_03_06/",
        f"/{RELATIVE_ROOT}/374_03_07/",
        f"/{RELATIVE_ROOT}/374_03_08/",
        f"/{RELATIVE_ROOT}/374_03_09/",
        
        # Packing of Randomly Placed Overlapping Spheres - Shale Reconstructions
        f"/{RELATIVE_ROOT}/374_04_00/",
        f"/{RELATIVE_ROOT}/374_04_01/", 
        
        # Complex fractures - Fractures with Permeable Walls
        f"/{RELATIVE_ROOT}/374_05_00/",
        f"/{RELATIVE_ROOT}/374_05_01/",
        f"/{RELATIVE_ROOT}/374_05_02/",
        f"/{RELATIVE_ROOT}/374_05_03/",
        
        # Yings pack - Porosity Gradient
        f"/{RELATIVE_ROOT}/374_06_00/",
        f"/{RELATIVE_ROOT}/374_06_01/",
        
        # Random - More Packs of Randomly Placed Overlapping Spheres
        f"/{RELATIVE_ROOT}/374_07_00/",
        f"/{RELATIVE_ROOT}/374_07_01/",
        f"/{RELATIVE_ROOT}/374_07_02/",
        
        # Realistic fractures
        f"/{RELATIVE_ROOT}/374_08_00/",

        # Fake cracs - Fractured Packing of Randomly Placed Overlapping Spheres
        f"/{RELATIVE_ROOT}/374_09_01/",
        f"/{RELATIVE_ROOT}/374_09_02/",
        f"/{RELATIVE_ROOT}/374_09_03/",
        f"/{RELATIVE_ROOT}/374_09_04/",
        
        # Not documented
        f"/{RELATIVE_ROOT}/374_10_01/",
        f"/{RELATIVE_ROOT}/374_10_02/",
        f"/{RELATIVE_ROOT}/374_10_03/",
        
        # BigSpheresPack
        f"/{RELATIVE_ROOT}/204_03/",
        f"/{RELATIVE_ROOT}/204_05/",
        f"/{RELATIVE_ROOT}/204_06/", 
        f"/{RELATIVE_ROOT}/204_07/", 
        f"/{RELATIVE_ROOT}/204_08/",
        f"/{RELATIVE_ROOT}/204_09/",
        f"/{RELATIVE_ROOT}/204_01/",
        f"/{RELATIVE_ROOT}/204_010/",
        f"/{RELATIVE_ROOT}/204_011/",
        f"/{RELATIVE_ROOT}/204_012/",
    
        # Rock: Estaillades Carbonate
        f"/{RELATIVE_ROOT}10_01/",
        
        # Rock: Estaillades Castlegate Sandstone and Gambier Limestone
        f"/{RELATIVE_ROOT}/16_02/",
        f"/{RELATIVE_ROOT}/16_01/",
    
        # Rock: Artificially Induced Fracture in Berea Sandstone
        f"/{RELATIVE_ROOT}/31_01/",
        f"/{RELATIVE_ROOT}/31_02/",
        f"/{RELATIVE_ROOT}/31_03/",
        f"/{RELATIVE_ROOT}/31_04/",
        
        # Rock: Fontainebleau
        f"/{RELATIVE_ROOT}/57_07/",
        f"/{RELATIVE_ROOT}/57_06/",
        f"/{RELATIVE_ROOT}/57_05/",
        f"/{RELATIVE_ROOT}/57_04/",
        f"/{RELATIVE_ROOT}/57_03/",
        f"/{RELATIVE_ROOT}/57_02/",
        f"/{RELATIVE_ROOT}/57_01/",
        
        # Estaillades Carbonate
        f"/{RELATIVE_ROOT}/58_01/",
        
        # Solid Comprised of regular grains
        f"/{RELATIVE_ROOT}/65_01/", 
        f"/{RELATIVE_ROOT}/65_02/", 
        f"/{RELATIVE_ROOT}/65_03/", 
        f"/{RELATIVE_ROOT}/65_04/", 
        f"/{RELATIVE_ROOT}/65_05/", 
        
        # Statiscally Generated Medium
        f"/{RELATIVE_ROOT}/69_01/", 
        
        # Savonnières carbonate
        f"/{RELATIVE_ROOT}/72_01/", 
        f"/{RELATIVE_ROOT}/72_02/", 
        
        # Massangis Jaune carbonate
        f"/{RELATIVE_ROOT}/73_01/", 
        f"/{RELATIVE_ROOT}/73_02/", 
        
        # Bentheimer, Berea and Leopard Sandstones
        f"/{RELATIVE_ROOT}/135_00/",
        f"/{RELATIVE_ROOT}/135_02/",
        f"/{RELATIVE_ROOT}/135_03/",             
        f"/{RELATIVE_ROOT}/135_04/",


        
        # Bidisperse sphere packs generated under gravity
        f"/{RELATIVE_ROOT}/204_02/",
        f"/{RELATIVE_ROOT}/204_04/", 
        
        # Clay geometry
        f"/{RELATIVE_ROOT}/276_01/",  # Comentado no codigo original

        # Belgian Fieldstone
        f"/{RELATIVE_ROOT}/297_01/",   # Comentado no codigo original
        
        # Sandstones: Bandera Brown, Bandera Gray, Bentheimer, Berea, 
        # Berea Sister Gray, Berea Upper Gray, Buff Berea, CastleGate, Kirby,
        # Leopard, Parker
        f"/{RELATIVE_ROOT}/317_01/",
        f"/{RELATIVE_ROOT}/317_02/",  # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_03/",  # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_04/",  # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_05/",  # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_06/",  # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_07/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_08/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_09/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_010/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/317_011/", # Comentado no codigo original

        # Rough fractures
        f"/{RELATIVE_ROOT}/339_01/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/339_02/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/339_03/", # Comentado no codigo original
        
        # Bentheimer
        f"/{RELATIVE_ROOT}/344_02/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/344_03/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/344_04/", # Comentado no codigo original
        f"/{RELATIVE_ROOT}/344_05/", # Comentado no codigo original
        
        # Vaca Muerta 
        f"/{RELATIVE_ROOT}/207_01/", # Nao aparece no codigo original
        f"/{RELATIVE_ROOT}/207_02/", # Nao aparece no codigo original
        f"/{RELATIVE_ROOT}/207_03/", # Nao aparece no codigo original
        f"/{RELATIVE_ROOT}/207_04/", # Nao aparece no codigo original
        """
        Group_0 = [ 
            # Modified FinneyPacks
            f"/{RELATIVE_ROOT}/374_01_00/", 
            f"/{RELATIVE_ROOT}/374_01_01/",
            f"/{RELATIVE_ROOT}/374_01_02/",
            f"/{RELATIVE_ROOT}/374_01_03/",
            f"/{RELATIVE_ROOT}/374_01_04/",
            f"/{RELATIVE_ROOT}/374_01_05/",
            f"/{RELATIVE_ROOT}/374_01_06/",
            f"/{RELATIVE_ROOT}/374_01_07/",
        ]
        
        Group_1 = [
            # Bentheimer
            f"/{RELATIVE_ROOT}/172_01/",
            f"/{RELATIVE_ROOT}/172_02/",
            f"/{RELATIVE_ROOT}/172_03/",
        ]
        paths = []
        paths_480 = []
        
        for path in Group_1:
            path = path.replace("//", '/')
            
            main_name       = [part for part in path.split("/") if part][-1]
            main_name_256   = main_name+"_256"
    
            #final_name_256_mat = f"{path}{main_name_256}/{main_name_256}.mat"
            final_name_256_mat = f"{path}{main_name_256}.mat"
            for pressure_path in ["P_1_MPa.mat", "P_2_MPa.mat","P_5_MPa.mat","P_10_MPa.mat","P_20_MPa.mat"]:
                
                try:
                    final_name_256_mat = self.resolve_mat_path(final_name_256_mat)
                    try:
                        final_name_256_MPA = f"{path}{pressure_path}"
                        final_name_256_MPA = self.resolve_mat_path(final_name_256_MPA)
                        paths.append( (final_name_256_mat, final_name_256_MPA))
                    except Exception as e:
                        
                        print(f"Pressure {pressure_path} .mat not found: ", main_name_256, f"\n - {final_name_256_mat}\n - {final_name_256_MPA}")
                        continue
                except:
                    print("Main .mat not found: ", main_name_256)
                    continue
        
        return paths, paths_480