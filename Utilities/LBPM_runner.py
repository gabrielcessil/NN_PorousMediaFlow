import os
import subprocess
import re
from Utilities import domain_creator as dc
from Utilities.plotter import plot_heatmap

def get_vti_files(directory, full_path=False):
    """
    Retorna uma lista com os nomes ou caminhos completos de todos os arquivos .vti
    presentes no diretório especificado.

    Parameters:
    - directory (str): Caminho do diretório onde procurar pelos arquivos .vti.
    - full_path (bool): Se True, retorna o caminho completo dos arquivos.

    Returns:
    - list: Lista com os nomes ou caminhos completos dos arquivos .vti encontrados.
    """
    try:
        # Filtrar arquivos com extensão .vti no diretório fornecido
        vti_files = [
            os.path.join(directory, file) if full_path else file
            for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file)) and file.endswith(".vti")
        ]
        if len(vti_files) == 0: 
            print("Nenhum arquivo .vti encontrado no diretório ", directory)
            return []
            
        if len(vti_files) != 1: 
            raise Exception("More than one .vti file inside",directory,": ", vti_files)

        return vti_files[0]
    except FileNotFoundError:
        print(f"Erro: O diretório '{directory}' não foi encontrado.")
        return []
    except PermissionError:
        print(f"Erro: Permissão negada para acessar o diretório '{directory}'.")
        return []
        
def get_raw_files(directory, full_path=False):
    """
    Retorna uma lista com os nomes ou caminhos completos de todos os arquivos .vti
    presentes no diretório especificado.

    Parameters:
    - directory (str): Caminho do diretório onde procurar pelos arquivos .vti.
    - full_path (bool): Se True, retorna o caminho completo dos arquivos.

    Returns:
    - list: Lista com os nomes ou caminhos completos dos arquivos .vti encontrados.
    """
    try:
        # Filtrar arquivos com extensão .vti no diretório fornecido
        raw_files = [
            os.path.join(directory, file) if full_path else file
            for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file)) and file.endswith(".raw")
        ]
        
        if len(raw_files) != 1: raise Exception(f"More than one .raw file inside {directory}: {raw_files}")

        return raw_files[0]
    except FileNotFoundError:
        print(f"Erro: O diretório '{directory}' não foi encontrado.")
        return []
    except PermissionError:
        print(f"Erro: Permissão negada para acessar o diretório '{directory}'.")
        return []
    
def get_folder_names(directory, full_path=False):
    """
    Retorna uma lista contendo os nomes das pastas presentes no diretório especificado.
    Se full_path for True, retorna o caminho completo das pastas.

    Parameters:
    - directory (str): Caminho do diretório onde as pastas serão listadas.
    - full_path (bool): Se True, retorna o caminho completo das pastas.

    Returns:
    - list: Lista com os nomes ou caminhos completos das pastas.
    """
    try:
        # Filtrar apenas os diretórios na lista do conteúdo do diretório fornecido
        folder_names = [
            os.path.join(directory, folder) if full_path else folder
            for folder in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, folder))
        ]
        return folder_names
    except FileNotFoundError:
        print(f"Erro: O diretório '{directory}' não foi encontrado.")
        return []
    except PermissionError:
        print(f"Erro: Permissão negada para acessar o diretório '{directory}'.")
        return []
    

def find_highest_vis_folder(base_folder):
    """
    Finds the folder with the highest numeric suffix in the base_folder.

    Parameters:
    - base_folder (str): The path to the folder containing 'vis+number' directories.

    Returns:
    - str: The full path to the folder with the highest numeric suffix.
    """
    vis_folders = []
    pattern = re.compile(r"^vis(\d+)$")  # Matches folders with the format 'vis<number>'
    
    # Scan for matching folders
    for folder in os.listdir(base_folder):
        match = pattern.match(folder)
        if match:
            vis_folders.append((int(match.group(1)), folder))
    
    # Get the folder with the highest number
    if vis_folders:
        highest_folder = max(vis_folders, key=lambda x: x[0])[1]
        return base_folder+"/"+highest_folder+"/"
    else:
        raise FileNotFoundError("No 'vis+number' folders found in the specified directory. Ensure LBPM runned succesfully")

    
def folder_Creator(diretorio, nome_pasta):
    """
    Cria uma pasta com o nome especificado em um diretório.

    Parâmetros:
    - diretorio (str): O caminho do diretório onde a pasta será criada.
    - nome_pasta (str): O nome da pasta a ser criada.

    Retorna:
    - str: Caminho completo da pasta criada ou existente.
    """
    # Cria o caminho completo
    caminho_completo = os.path.join(diretorio, nome_pasta)
    
    # Verifica se a pasta já existe
    if not os.path.exists(caminho_completo):
        os.makedirs(caminho_completo)
        print(f"Pasta '{nome_pasta}' criada em: {diretorio}")
    else:
        print(f"A pasta '{nome_pasta}' já existe em: {diretorio}")
    
    return caminho_completo

class db_Creator:
    def __init__(self,
        tau = 1.0, # Relaxation
        F = [1.0e-7, 0.0, 0.0], # Interface Forces
        timestepMax = 20000, # Time steps
        tolerance = 0.0001, # Max difference in feature readings to stop simulation
        read_type = "8bit", # data type
        N = [512, 256, 1], # X Y Z : size of original image
        nproc = [1, 1, 1], # process grid (cores)
        n = [512, 256, 1], # X Y Z : sub-domain size
        offset = [0, 0, 0], # offset to read sub-domain
        voxel_length = 1.0, # voxel length (in microns)
        read_values = [0, 1, 2], # labels within the original image 
        write_values = [0, 1, 1], # associated labels to be used by LBPM   (0:solid, 1..N:fluids)
        boundary_condition = 0, # boundary condition type (0 for periodic)
        write_silo = True, # write SILO databases with assigned variables
        save_8bit_raw = True, # write labeled 8-bit binary files with phase assignments
        save_phase_field = True, # save phase field within SILO database
        save_pressure = True, # save pressure field within SILO database
        save_velocity = True, # save velocity field within SILO database
        ):
        
        self.tau = tau
        self.F = F
        self.timestepMax = timestepMax
        self.tolerance = tolerance
        self.read_type = read_type
        self.N = N
        self.nproc = nproc
        self.n = n
        self.offset = offset
        self.voxel_length = voxel_length
        self.read_values = read_values
        self.write_values = write_values
        self.boundary_condition = boundary_condition
        self.write_silo = write_silo
        self.save_8bit_raw = save_8bit_raw
        self.save_phase_field = save_phase_field
        self.save_pressure = save_pressure
        self.save_velocity = save_velocity
        
    def _get_Text(self, raw_filename):
        return f"""
                MRT {{
                   tau = {self.tau}
                   F = {', '.join(map(str, self.F))}
                   timestepMax = {self.timestepMax}
                   tolerance = {self.tolerance}
                }}
                Domain {{
                   Filename = "{raw_filename}"
                   ReadType = "{self.read_type}"      // data type
                   N = {', '.join(map(str, self.N))}     // size of original image
                   nproc = {', '.join(map(str, self.nproc))}        // process grid
                   n = {', '.join(map(str, self.n))}      // sub-domain size
                   offset = {', '.join(map(str, self.offset))} // offset to read sub-domain
                   voxel_length = {self.voxel_length}    // self.voxel length (in microns)
                   ReadValues = {', '.join(map(str, self.read_values))}   // labels within the original image 
                   WriteValues = {', '.join(map(str, self.write_values))} // associated labels to be used by LBPM   (0:solid, 1..N:fluids)
                   BC = {self.boundary_condition}                 // boundary condition type (0 for periodic)
                }}
                Visualization {{
                   write_silo = {str(self.write_silo).lower()}     // write SILO databases with assigned variables
                   save_8bit_raw = {str(self.save_8bit_raw).lower()}  // write labeled 8-bit binary files with phase assignments
                   save_phase_field = {str(self.save_phase_field).lower()}  // save phase field within SILO database
                   save_pressure = {str(self.save_pressure).lower()}    // save pressure field within SILO database
                   save_velocity = {str(self.save_velocity).lower()}    // save velocity field within SILO database
                }}"""
                
    def Create_File(self, folder, raw_filename):
        # Get the text content for the .db file
        db_file_content = self._get_Text(raw_filename + ".raw")
        
        # Construct the full path for the file
        full_file_path = os.path.join(folder, raw_filename + ".db")
        
        # Write the file to the specified folder
        with open(full_file_path, 'w+') as arquivo:
            arquivo.write(db_file_content)
        print(f"Arquivo '{full_file_path}' criado com sucesso.")


def run_commands_in_directory(commands):
    """
    Execute a list of commands sequentially in specific directories.

    Parameters:
    - commands (dict): A dictionary where keys are directories and values are commands to run.

    Returns:
    - None
    """
    for directory, command in commands.items():
        try:
            print(f"Running: {command} in {directory}")
            result = subprocess.run(
                command,
                cwd=directory,
                shell=True,  # Enables running shell commands
                text=True,  # Decodes output into strings
                check=True,  # Raises an exception if a command fails
                capture_output=True,  # Captures stdout and stderr
            )
            print(f"Output:\n{result.stdout.strip()}")
            if result.stderr:
                print(f"Error Output:\n{result.stderr.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")
            print(f"Error: {e.stderr.strip()}")
            break  # Stop execution if a command fails

def Run_Example(simulations_main_folder,
                 simulation_name,
                 lbm_file_path,
                 lbm_functional="lbpm_permeability_simulator",
                 mpi_file_path=None,
                 vis2vtk=True):
    """
    Main function to run the LBM simulation and convert SILO files.

    Parameters:
    - simulations_main_folder (str): Path to the main simulations folder.
    - simulation_name (str): Name of the simulation.
    - file_path_silo2vti_exe (str): Path to the Silo2VTI executable.
    - lbm_file_path (str): Path to the LBM installation folder.
    - lbm_functional (str): LBM functional executable name.

    Returns:
    - None
    """
    # Join the simulation folder
    file_path_simulation = os.path.join(simulations_main_folder, simulation_name)

    # Run LBPM: vis folder will be created
    if mpi_file_path is None: # Assume same path for MPI
        lbm_command = f"{lbm_file_path}mpi/bin/mpirun -np 1 {lbm_file_path}LBPM_dir/tests/{lbm_functional} {simulation_name}.db"
    else:
        lbm_command = f"{mpi_file_path}bin/mpirun -np 1 {lbm_file_path}LBPM_dir/tests/{lbm_functional} {simulation_name}.db"
    run_commands_in_directory({file_path_simulation: lbm_command})
    
    # If the conversion from silo to vti is needed
    if vis2vtk:
        # Join the right vis folder
        vis_folder_path = find_highest_vis_folder(file_path_simulation)
        vis_folder = os.path.join(file_path_simulation, vis_folder_path)
        # Convert silo output to vti
        silo2vti_command = f"{lbm_file_path}converter_silo_vti/silo2vti summary.silo output.pvti"
        run_commands_in_directory({vis_folder: silo2vti_command})
    

def Save_Example(domain, simulation_name, simulations_main_folder, plot=False):
    # Create folder
    folder_name = folder_Creator(simulations_main_folder, simulation_name)
    # Create .raw of this domain inside folder
    dc.save_as_raw(domain, folder_name, simulation_name)
    if domain.ndim == 2:
        # Create db file for LBPM run associated to this domain inside folder
        dbCr = db_Creator(N=[domain.shape[1], domain.shape[0], 1],
                          n=[domain.shape[1], domain.shape[0], 1])
    elif domain.ndim == 3:
        dbCr = db_Creator(N=[domain.shape[2], domain.shape[1], domain.shape[0]],
                          n=[domain.shape[2], domain.shape[1], domain.shape[0]])

    dbCr.Create_File(folder_name, simulation_name)
    # Create guiding image inside the folder
    if plot: plot_heatmap(domain, folder_name, simulation_name)


