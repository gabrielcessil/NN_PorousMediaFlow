import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import matplotlib
import matplotlib as mpl
from PIL import Image
from skimage.measure import regionprops, label
import seaborn as sns
from scipy.stats import skew
import matplotlib.gridspec as gridspec
import math
import os
import pyvista as pv
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


def Plot_Streamlines(u, v, w, domain, filename, mask=None, seed_stride=4, max_time=1000.0, use_tubes=False):
    """
    Plots streamlines from a 3D vector field and saves as PNG.

    Parameters:
        u, v, w (np.ndarray): 3D arrays of velocity components.
        domain (np.ndarray): 3D binary array (1=solid, 0=fluid) defining the domain.
        filename (str): Output file path (without extension).
        mask (np.ndarray): Optional boolean mask to restrict domain (not used in this version).
        seed_stride (int): Step between seed points (larger = fewer).
        max_time (float): Max integration time for streamlines.
        use_tubes (bool): If True, render streamlines as tubes.
    """
    # Ensure output folder exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # Downsample vector fields
    u = u[::seed_stride, ::seed_stride, ::seed_stride]
    v = v[::seed_stride, ::seed_stride, ::seed_stride]
    w = w[::seed_stride, ::seed_stride, ::seed_stride]
    shape = u.shape

    # Create grid coordinates
    x, y, z = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing='ij'
    )

    # Create structured grid with vectors
    grid = pv.StructuredGrid(x, y, z)
    vectors = np.stack((u, v, w), axis=-1)
    grid["vectors"] = vectors.reshape(-1, 3)

    # Create streamlines
    streamlines, src = grid.streamlines(
        return_source=True,
        max_time=max_time,
        initial_step_length=2.0,
        terminal_speed=0.1,
        n_points=25,
        source_radius=2.0,
        source_center=(shape[0] // 2, shape[1] // 2, shape[2] // 2),
    )

    # Setup plotter
    p = pv.Plotter(off_screen=True)
    p.add_mesh(grid.outline(), color='k')

    # Add semi-transparent domain contour (if provided)
    if domain is not None:
        domain_ds = domain[::seed_stride, ::seed_stride, ::seed_stride].astype(np.float32)
        domain_grid = pv.StructuredGrid(x, y, z)
        domain_grid["domain"] = domain_ds.flatten(order="F")
        try:
            contour = domain_grid.contour([0.5])
            p.add_mesh(contour.extract_all_edges(), color='grey', opacity=0.25)
        except Exception as e:
            print("Warning: Failed to extract contour from domain:", e)

    # Add streamlines
    if use_tubes:
        p.add_mesh(streamlines.tube(radius=0.15), color="dodgerblue")
    else:
        p.add_mesh(streamlines, color="dodgerblue")

    # Add streamline seed source
    p.add_mesh(src, color="red", opacity=0.4)
    p.add_axes()
    p.show_bounds(grid="back", location="outer", font_size=12)
    """
    p.camera_position = [
        (shape[0] * 1.5, shape[1] * 1.5, shape[2] * 0.8), 
        (shape[0] // 2, shape[1] // 2, shape[2] // 2), 
        (0, 0, 1)
    ]
    """
    # Save screenshot
    p.screenshot(filename + ".png")



def Plot_Domain(values, filename, remove_value=[], colormap='cool', special_colors={}):
    """
    Parameters:
        values (np.ndarray): 3D NumPy array of cell values.
        filename (str): Name of the output file (with path, without extension).
        remove_value (list): List of values to mark as ghost cells (optional).
        colormap (str): Colormap for non-zero and non-one values (default: 'cool').
        clim (list): Color range limits [min, max] for the colormap (default: [0, 255]).
    """

    # Ensure the folder for the output file exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # Create structured grid (ImageData)
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1  # Dimensions as points
    grid.origin = (0, 0, 0)  # Origin of the grid
    grid.spacing = (1, 1, 1)  # Uniform spacing

    # Assign cell values
    grid.cell_data["values"] = values.flatten(order="F")
    
    # Convert to an unstructured grid for filtering
    mesh = grid.cast_to_unstructured_grid()

    # Remove unwanted cells: OK
    for removed_value in remove_value:
        to_remove_mask = np.argwhere(mesh["values"] == removed_value)
        mesh.remove_cells(to_remove_mask.flatten(), inplace=True)

    # Separate different cell types
    special_cells = {}
    for value, color in special_colors.items():
        special_cells[value] = (mesh.extract_cells(np.where(mesh["values"] == value)[0]), color)

    
    # Configure the plotter
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)

    # Plot other cells with the colormap
    if mesh.n_cells > 0:
        
        plotter.add_mesh(
            mesh,
            cmap=colormap,
            show_edges=False,
            lighting=True,
            smooth_shading=False,
            split_sharp_edges=False,
            scalar_bar_args={
                "title": "Continuous Range",
                "vertical": True,
                "title_font_size": 40,
                "label_font_size": 25,
                "position_x": 0.8,
                "position_y": 0.05,
                "height": 0.9,
                "width": 0.05,
                "n_labels": 10,
            },
        )
        
    for value, (cells, color) in special_cells.items():
        if cells.n_cells > 0:
            plotter.add_mesh(cells, color=color, show_scalar_bar=False)

    # Add axis indicators
    plotter.add_axes(line_width=5, cone_radius=0.6, shaft_length=0.9, tip_length=0.2, ambient=0.5, label_size=(0.25, 0.15))

    # Show grid bounds with labels
    plotter.show_bounds(
        grid='back', location='outer', ticks='both',
        show_xlabels=True, show_ylabels=True, show_zlabels=True,
        n_xlabels=4, n_ylabels=4, n_zlabels=4,
        font_size=15, xtitle='x', ytitle='y', ztitle='z'
    )

    plotter.screenshot(filename + ".png")
    plotter.close()
    del plotter

def Plot_Classified_Domain(values, filename, remove_value=[], labels={}, colormap='cool', show_label=True, special_colors={}):
    # Ensure the output directory exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # CREATE GRID STRUCTURE
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1  # Adjust for point-based representation
    grid.origin = (0, 0, 0)  # Set grid origin
    grid.spacing = (1, 1, 1)  # Uniform voxel spacing



    # MAP ORIGINAL VALUES
    unique_classes = np.unique(values) 
    discrete_linspace_values = np.round(np.linspace(0, 1, len(unique_classes)+1), 4)
    #step = (discrete_linspace_values[1]-discrete_linspace_values[0]
    normalized_mapping = {key: val + 0.0001 for key, val in zip(unique_classes, discrete_linspace_values)}

    denormalization_mapping = {v: k for k, v in normalized_mapping.items()}
    
    
    # ASSING VALUES TO MESH STRUCTURE
    mesh_normalized_values = np.round(np.vectorize(normalized_mapping.get)(values), 4)
    grid.cell_data["class_values"] = mesh_normalized_values.flatten(order="F")
    mesh = grid.cast_to_unstructured_grid()


    # REMOVE UNWATED CELLS
    for removed_value in remove_value:
        if removed_value in normalized_mapping:
            normalized_value_to_remove = normalized_mapping[removed_value]
            to_remove_mask = np.argwhere(mesh["class_values"] == normalized_value_to_remove)
            mesh.remove_cells(to_remove_mask.flatten(), inplace=True)
            
        
    # Unique values in mesh
    mesh_unique_values = np.unique(mesh["class_values"])
    num_mesh_unique_values = len(mesh_unique_values)
    
    
    # MANAGE COLORS
    # Get colormap for custom labels
    default_class_colors = {}
    for special_value, special_color in special_colors.items():
        if special_value in normalized_mapping:
            default_class_colors[ normalized_mapping[special_value] ] = special_color
    # Get colormap from values remaining in mesh
    nonDefault_classes = []
    for value in mesh_unique_values:
        if not (value in default_class_colors.keys()):
            nonDefault_classes.append(value)
    num_nonDefault_classes = len(nonDefault_classes)
    # Make colormap with custom colors
    if num_nonDefault_classes>0:
        colormap = mpl.colormaps[colormap].resampled(num_nonDefault_classes)
        color_space = np.linspace(1, 0, num_nonDefault_classes+1)[0:-1]
        if num_nonDefault_classes>1:
            color_space = color_space + (color_space[1]-color_space[0])/2
        generated_colors = colormap(color_space)
    color_list = []
    color_index = 0 # Avoid extreme colors (ex:too dark)  
    for val in mesh_unique_values:
        if val in default_class_colors:  # Use predefined colors for 0 and 1
            color = default_class_colors[val]
            color_list.append(mcolors.to_hex(color, keep_alpha=True))
        else:
            color = generated_colors[color_index] # Get color from colormap
            color_list.append(mcolors.to_hex(color, keep_alpha=True))
            color_index +=1
    
    # MAKE CUSTOM TICKS
    # Evenly distribute tick locations to avoid overlap, even if values are far apart
    if num_nonDefault_classes > 1:
        tick_positions = np.linspace(0, 1, num_mesh_unique_values+1)[0:-1]
        tick_positions = tick_positions + (tick_positions[1]-tick_positions[0])/2 # Set position to the middle of the range
    else:
        tick_positions = [discrete_linspace_values[0]]
    # Create annotation mapping with proper spacing
    annotations = {}
    for tick, val in zip(tick_positions, mesh_unique_values):
        original_val = denormalization_mapping[val]
                
        if int(denormalization_mapping[val]) in labels or float(denormalization_mapping[val]) in labels:
            annotations[tick] = f"            {labels[original_val]}" 
        else:
            annotations[tick] = f"            {original_val}"

    
    # MAKE THE PLOT
    plotter = pv.Plotter(window_size=[1920, 1080])  # Full HD resolution
    if mesh.n_cells > 0:
        plotter.add_mesh(
            mesh,
            scalars=mesh["class_values"],
            categories=True,
            cmap= color_list,
            show_edges=False,
            lighting=True,
            smooth_shading=False,
            split_sharp_edges=False,
            annotations = annotations,
            scalar_bar_args={
                "title": "    Classes",
                "vertical": True,
                "title_font_size": 40,
                "label_font_size": 25,
                "position_x": 0.8,
                "position_y": 0.05,
                "height": 0.9,
                "width": 0.05,
                "n_labels": 0,
            },
            clim=[0, 1] # Influences the color setting
        )
        
    # Add axis indicators
    plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.9,
        tip_length=0.2,
        ambient=0.5,
        label_size=(0.25, 0.15)
    )

    # Show grid bounds
    plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=4,
        font_size=15,
        xtitle='x',
        ytitle='y',
        ztitle='z'
    )

    # Show the visualization
    plotter.show()

    # Save the visualization as an image
    plotter.screenshot(filename + ".png")
    
    return filename + ".png"




def plot_hist(data, bins=30, title='Histogram', filename=None, notable=[], xlim=(), ylim=(), color='blue'):
    
    num_main_categories = len(data)
    
    fig, axes = plt.subplots(num_main_categories, 1, figsize=(15, 5 * num_main_categories))
    if num_main_categories == 1:
        axes = [axes]
    
    plt.subplots_adjust(top=0.9)  # Adjust layout to prevent title overlap
    plt.suptitle(title, fontsize=16, y=0.98)  # Move the title slightly upwards
    
    
    # Iterate over data_dict and check each element type
    for i, (category, content) in enumerate(data.items()):
        
        # Single histogram
        axes[i].hist(content, bins=bins, edgecolor='black', alpha=0.5, color=color, density=True, label=category)
        
        axes[i].set_title(f'{category}')
        for x_value in notable:
            axes[i].axvline(x_value, color='k', linestyle='dashed', linewidth=1, label='Notable' if 'Notable' not in axes[i].get_legend_handles_labels()[1] else None)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        if xlim: axes[i].set_xlim(xlim)
        if ylim: axes[i].set_ylim(ylim)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save or show the plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
    else:
        plt.show()
        
def Plot_Classified_Domain_2D(values, filename, remove_value=[], labels={}, colormap='cool', show_label=True, special_colors={}):
    # Ensure the output directory exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # If input is 3D (1, H, W), take first slice
    if len(values.shape) == 3 and values.shape[0] == 1:
        values = values[0]

    # Get unique classes (ignoring remove_value)
    unique_classes = np.unique(values)
    unique_classes = unique_classes[~np.isin(unique_classes, remove_value)]

    expoente = math.floor(math.log10((1/(len(unique_classes)-1)))) if len(unique_classes) > 1 else 0
    delta = 10 ** expoente if len(unique_classes) > 1 else 1
    n_decimals = np.abs(expoente)+1 if len(unique_classes) > 1 else 1
    discrete_linspace_values = np.linspace(start=0, stop=1, num=len(unique_classes))
    normalized_mapping = {class_i: round(norm_val, n_decimals) for class_i, norm_val in zip(unique_classes, discrete_linspace_values)}
    denormalization_mapping = {v: round(k, n_decimals) for k, v in normalized_mapping.items()}

    # Normalize values
    normalized_values = np.vectorize(lambda x: normalized_mapping.get(x, np.nan))(values)

    # Handle special colors
    default_class_colors = {}
    for special_value, special_color in special_colors.items():
        if special_value in normalized_mapping:
            default_class_colors[normalized_mapping[special_value]] = special_color

    # Generate colors for remaining classes
    nonDefault_classes = [val for val in normalized_mapping.values() if val not in default_class_colors]
    num_nonDefault_classes = len(nonDefault_classes)
    if num_nonDefault_classes > 0:
        cmap = mpl.colormaps[colormap].resampled(num_nonDefault_classes)
        color_space = np.linspace(1, 0, num_nonDefault_classes+1)[0:-1]
        if num_nonDefault_classes > 1:
            color_space = color_space + (color_space[1]-color_space[0])/2
        generated_colors = cmap(color_space)

    # Build final color list
    color_list = []
    legend_elements = []
    color_index = 0
    for val in sorted(normalized_mapping.values()):
        if val in default_class_colors:
            color = default_class_colors[val]
        else:
            color = generated_colors[color_index]
            color_index += 1
        color_list.append(color)

        if show_label:
            original_val = denormalization_mapping[val]
            label_text = labels.get(original_val, str(original_val))
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label_text))

    # Create custom colormap
    cmap = mcolors.ListedColormap(color_list)
    bounds = np.linspace(0, 1, len(color_list)+1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(normalized_values, cmap=cmap, norm=norm, interpolation='none')

    # Add legend instead of colorbar
    if show_label:
        ax.legend(handles=legend_elements, loc='upper left', fontsize='xx-large', frameon=True)


    ax.set_xticks([])           # Hide x-axis ticks
    ax.set_yticks([])           # Hide y-axis ticks
    ax.set_title("")            # Remove title
    ax.set_xlabel("")           # Remove x-axis label
    ax.set_ylabel("")           # Remove y-axis label
    
    fig.tight_layout()
    


    plt.savefig(filename + ".png", bbox_inches='tight', dpi=300)
    plt.close()
    return filename + ".png"


def Plot_Continuous_Domain_2D(
    values,
    filename,
    title="",
    remove_value=None,              # values to make transparent
    colormap="viridis",
    vmin=None,
    vmax=None,
    clip_percentiles=None,          # e.g. (2, 98) for robust scaling
    show_colorbar=True,
    special_colors=None,            # dict: {value: color}
    dpi=300
):
    """
    Save a PNG heatmap of a 2D continuous field, with optional special colors.

    Parameters
    ----------
    values : np.ndarray
        2D array (H, W) or (1, H, W) with continuous values.
    filename : str
        Path (without extension) to save the image.
    remove_value : float or list/tuple[float], optional
        Values to be masked (transparent).
    colormap : str
        Matplotlib colormap name for continuous values.
    vmin, vmax : float, optional
        Explicit color limits. If None, computed from data (or percentiles).
    clip_percentiles : tuple, optional
        If vmin/vmax not given, use percentiles of data.
    show_colorbar : bool
        Whether to show colorbar.
    special_colors : dict, optional
        Mapping {value: matplotlib_color} for specific discrete values.
    dpi : int
        Output resolution.
    """
    import matplotlib.pyplot as plt

    # Ensure output directory exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # If input is 3D (1, H, W), take first slice
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"`values` must be 2D or (1,H,W). Got {values.shape}")

    # Build mask: NaNs + remove_value(s)
    mask = np.isnan(values)
    if remove_value is not None:
        if np.isscalar(remove_value):
            mask |= (values == remove_value)
        else:
            mask |= np.isin(values, list(remove_value))

    # Prepare data for plotting
    data = np.ma.masked_array(values, mask=mask)

    # Determine vmin/vmax if not given
    finite_vals = data.compressed()
    if finite_vals.size == 0:
        raise ValueError("All values masked or NaN; nothing to plot.")
    if (vmin is None or vmax is None):
        if clip_percentiles is not None:
            lowp, highp = clip_percentiles
            vmin_auto, vmax_auto = np.percentile(finite_vals, [lowp, highp])
        else:
            vmin_auto, vmax_auto = np.min(finite_vals), np.max(finite_vals)
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
        if vmin == vmax:
            vmin, vmax = vmin - 1e-8, vmax + 1e-8

    # Base colormap
    cmap = mpl.colormaps[colormap].copy()
    if hasattr(cmap, "set_bad"):
        cmap.set_bad((0, 0, 0, 0))  # transparent for masked

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")

    # Overlay special colors
    if special_colors:
        for val, color in special_colors.items():
            mask_special = (values == val)
            if np.any(mask_special):
                overlay = np.zeros((*values.shape, 4))
                overlay[mask_special] = mcolors.to_rgba(color)
                ax.imshow(overlay, interpolation="none")

    # Aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize="large")

    fig.tight_layout()
    out_path = f"{filename}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path

    
def plot_heatmap(array_2d, output_file="heatmap_hd", dpi=300, cmap="inferno", xlabel="X-axis", ylabel="Y-axis", title="Heatmap", colorbar_label="Value", vmin=None, vmax=None, grid=False):

    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    # If the first or last dimension is 1, squeeze it
    if array_2d.ndim == 3:
        if array_2d.shape[0] == 1:  # Case: (1, H, W) -> (H, W)
            array_2d = array_2d.squeeze(0)
        elif array_2d.shape[-1] == 1:  # Case: (H, W, 1) -> (H, W)
            array_2d = array_2d.squeeze(-1)
        
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 9))  # Full HD aspect ratio

    # Plot the heatmap with no interpolation (solid colors) and custom color range
    if vmin is None and vmax is None:
        heatmap = ax.imshow(array_2d, cmap=cmap, aspect="auto",
                        origin="upper", interpolation="none")
    else:
        heatmap = ax.imshow(array_2d, cmap=cmap, aspect="auto",
                        origin="upper", interpolation="none", vmin=vmin, vmax=vmax)

    # Add gridlines for the grid view
    if grid:
        ax.set_xticks(np.arange(-0.5, array_2d.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, array_2d.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(colorbar_label)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save the plot in HD resolution
    plt.savefig(output_file+".png", dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_classified_map(array_2d, output_file="classified_map", dpi=300, xlabel="X-axis", ylabel="Y-axis",
                        title="Classified Map", colorbar_label="Classes", grid=False, colormap='cool'):
    """
    Create and save a classified color map from a 2D NumPy array with distinct solid colors per class.

    Parameters:
        array_2d (np.ndarray): 2D NumPy array containing classified values.
        output_file (str): File name to save the classified map.
        dpi (int): Resolution of the output image.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        colorbar_label (str): Label for the colorbar.
        grid (bool): Whether to show a grid overlay.
    """
    
    # Ensure output directory exists
    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # Ensure the array is 2D
    if array_2d.ndim == 3:
        if array_2d.shape[0] == 1:
            array_2d = array_2d.squeeze(0)
        elif array_2d.shape[-1] == 1:
            array_2d = array_2d.squeeze(-1)

    # Get unique class values
    unique_values = np.unique(array_2d)
    unique_values = np.array(sorted(unique_values, key=lambda x: (x not in [0, 1], x)))

    non_default_values = unique_values[~np.isin(unique_values, [0, 1])]
    num_classes = len(non_default_values)
    
    if num_classes > 0:
        cmap = plt.cm.get_cmap(colormap, num_classes)
        colors = cmap(np.linspace(0, 1, num_classes))
    else:
        colors = []  # or set to None or default colors
        
        
    # Extract the colors
    generated_colors = list(colors)  # Convert to list to allow modifications
    
    # Define Grey for 0 and Black for 1
    default_class_colors = []
    
    if np.isin(unique_values, [0]).any():  
        default_class_colors.append((0.5, 0.5, 0.5, 1.0))  # Grey (for label 0)
    
    if np.isin(unique_values, [1]).any():  
        default_class_colors.append((0.0, 0.0, 0.0, 1.0))  # Black (for label 1)
        
    # Append the generated colors
    final_colors = default_class_colors + generated_colors  # Prepend grey and black
    
    listed_cmap = mcolors.ListedColormap(final_colors[:len(unique_values)])  # Ensure only the needed colors are used

    # Create a dictionary mapping class values to colorbar labels
    class_labels = {v: i for i, v in enumerate(unique_values)}

    # Replace array values with corresponding indices for correct color mapping
    indexed_array = np.vectorize(class_labels.get)(array_2d)

    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the classified data with discrete colors
    heatmap = ax.imshow(indexed_array, cmap=listed_cmap, aspect="auto",
                        origin="upper", interpolation="none")

    # Add gridlines if needed
    if grid:
        ax.set_xticks(np.arange(-0.5, array_2d.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, array_2d.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Create a discrete colorbar
    cbar = plt.colorbar(heatmap, ax=ax, ticks=range(len(unique_values)))
    cbar.set_label(colorbar_label)
    cbar.set_ticks(range(len(unique_values)))
    cbar.set_ticklabels(unique_values)  # Ensure correct labels match colors

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save and display the plot
    plt.savefig(output_file + ".png", dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()
    
    return output_file + ".png"


def create_gif_from_filenames(image_filenames, gif_filename, duration=200, loop=0, erase_plots=True):
    """
    Creates a GIF from a list of image filenames.

    Parameters:
        image_filenames (list): A list of filenames for the images to include in the GIF.
        gif_filename (str): Name of the output GIF file.
        duration (int): Duration of each frame in milliseconds.
        loop (int): Number of times to loop the GIF (0 for infinite).
        erase_plots (bool): If True, delete the individual image files after creating the GIF.
    """
    if not image_filenames:
        print("Error: No image filenames provided.")
        return

    images = [Image.open(file) for file in image_filenames]

    if not images:
        print("Error: Could not open any images from the provided filenames.")
        return
    print(image_filenames)
    images[0].save(gif_filename+".gif", save_all=True, append_images=images[1:], loop=loop, duration=duration)

    if erase_plots:
        for file in image_filenames:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")
    


def plot_labeled_clusters(cluster_image, labels, output_file="labeled_clusters"):
    """
    Overlay numerical labels on top of a colorized cluster image.

    Args:
        cluster_image (np.ndarray): 3D RGB image displaying clusters.
        labels (np.ndarray): 2D labeled segmentation array.
        output_file (str): Path to save the result.
    """
    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        
    # Ensure the array is 2D
    if cluster_image.ndim == 3:
        if cluster_image.shape[0] == 1:
            cluster_image = cluster_image.squeeze(0)
        elif cluster_image.shape[-1] == 1:
            cluster_image = cluster_image.squeeze(-1)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the cluster image
    ax.imshow(cluster_image, interpolation="nearest")
    
    # Extract region properties
    props = regionprops(label(labels))  # Get properties of labeled clusters

    # Overlay text labels at cluster centroids
    for region in props:
        centroid = region.centroid  # (y, x) coordinates
        ax.text(centroid[1], centroid[0], f"{region.label}", 
                color="white", fontsize=12, fontweight="bold", 
                ha="center", va="center", bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title and save
    ax.set_title("Labeled Clusters")
    plt.savefig(output_file+".png", dpi=300, bbox_inches="tight")
    plt.show()
    
    
def plot_mean_deviation(x, y_means, y_devs, title="Algorithm Performance",
                        xlabel="Samples Cells / Rock Surface Cells", ylabel="Accuracy",
                        filename="performance_plot.png"):
    """
    Plots the mean accuracy and its deviation as shaded area.

    Parameters:
    - y_means: List or numpy array of mean accuracy values (y-axis).
    - y_devs: List or numpy array of standard deviation values.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - filename: Filename to save the plot.
    """    
    if isinstance(y_means, list):
        y_means = np.array(y_means)
    if isinstance(y_devs, list):
        y_devs = np.array(y_devs)


    # Use a modern style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create figure with the specified window size (1920x1080 pixels)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080 = 19.2 x 10.8 inches at 100 dpi

    # Plot mean curve
    ax.plot(x, y_means, color='#1f77b4', linewidth=2.5, label="Mean Accuracy")

    # Fill shaded deviation
    ax.fill_between(x, y_means - y_devs, y_means + y_devs, color='#1f77b4', alpha=0.2, label="Deviation")

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Customizing ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add a legend
    ax.legend(fontsize=12, loc="lower right", frameon=True, shadow=True)

    # Show grid for clarity
    ax.grid(True, linestyle="--", alpha=0.6)

    # Save the figure in high resolution
    plt.savefig(filename+".png", dpi=300, bbox_inches="tight")  # High-quality PNG

    # Show plot
    plt.show()
    

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.colors as mcolors
from scipy.stats import mode
from matplotlib.ticker import AutoMinorLocator

def plot_distributions(dict_data, normalize=False, title="Distribution Comparison", save_path="plot.svg", pallete="husl"):
    """
    Compare distributions of multiple arrays using histograms with a uniform number of bins.
    
    Args:
        dict_data (dict): Dictionary of {name: array} pairs to plot.
        normalize (bool, optional): If True, normalizes the histograms to a probability
                                    density. Defaults to False.
        title (str, optional): The title for the plot. Defaults to "Distribution Comparison".
        save_path (str, optional): The path to save the plot. Defaults to "plot.png".
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette(pallete, n_colors=len(dict_data))
    all_data = []

    # Flatten and collect all data into a single array
    for array in dict_data.values():
        all_data.append(np.array(array).flatten())

    if not all_data:
        print("Error: dict_data is empty or contains no valid arrays.")
        return
        
    all_data_concatenated = np.concatenate(all_data)
    
    # Calculate the number of bins based on all data using the Freedman-Diaconis rule
    bins = np.histogram_bin_edges(all_data_concatenated, bins='fd')

    # Iterate through the data and plot histograms with the calculated bins
    for (name, array), color in zip(dict_data.items(), palette):
        array = np.array(array).flatten()
        if array.size == 0:
            print(f"Warning: '{name}' array is empty. Skipping.")
            continue
        
        # Calculate statistics for the legend
        mean = np.mean(array)
        std = np.std(array)
        
        # Plot the histogram with the uniform bin edges
        sns.histplot(
            x=array,
            ax=ax,
            label=f'{name} (μ={mean:.4f}, σ={std:.4f}',
            color=color,
            alpha=0.6,
            stat='probability' if normalize else 'count',
            bins=bins,
            kde=False,
            element='step'
        )

    # Final plot styling
    min_val = np.percentile(all_data_concatenated, 0.1)
    max_val = np.percentile(all_data_concatenated, 99.9)
    ax.set_xlim(min_val, max_val)

    ax.set_xlabel('Value', fontsize=14)
    ax.set_ylabel('Density' if normalize else 'Count', fontsize=14)
    ax.set_title(title, fontsize=16)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.legend(fontsize=12, loc='best')
    ax.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.7)
    ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=5))
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()



    
def plot_scatter_sampled(xdata, ydata, npoints=5000, xlabel="xdata", ylabel="ydata", title="Scatter Plot", simetric=True,  save_path = None):
    """
    Creates a 2D scatter plot with density coloring using a random sample of points.
    This method is suitable when you want a scatter plot but the full dataset is too large.

    Args:
        xdata (np.ndarray): A NumPy array of xdata values.
        ydata (np.ndarray): A NumPy array of ydata values.
        npoints (int): The number of random points to sample for the plot.
        save_path (Optional[str]): The path to save the plot file (e.g., 'my_plot.png').
                                   If None, the plot will be displayed on screen.
    """
    # --- 1. Flatten the Arrays ---
    x_flat = xdata.flatten()
    y_flat = ydata.flatten()

    # --- 2. Randomly Sample the Points ---
    # Use a fixed random seed for reproducibility.
    np.random.seed(42)
    total_points = len(x_flat)
    if npoints >= total_points:
        print(f"Sample size ({npoints}) is greater than or equal to total points ({total_points}). Plotting all points.")
        sample_indices = np.arange(total_points)
    else:
        sample_indices = np.random.choice(total_points, size=npoints, replace=False)

    x_sample = x_flat[sample_indices]
    y_sample = y_flat[sample_indices]

    # --- 3. Calculate Point Density using Gaussian KDE on the Sampled Data ---
    data_points = np.vstack([x_sample, y_sample])
    kde = gaussian_kde(data_points)
    density_values = kde(data_points)

    # --- 4. Sort Points by Density for Better Visualization ---
    sort_indices = density_values.argsort()
    x_sorted = x_sample[sort_indices]
    y_sorted = y_sample[sort_indices]
    density_sorted = density_values[sort_indices]

    # --- 5. Create the Plot ---
    fig, ax = plt.subplots(figsize=(12, 12))

    scatter = ax.scatter(x_sorted, y_sorted,
                         c=density_sorted,
                         cmap='plasma',
                         s=25,
                         alpha=0.6)

    # Add a color bar and increase its label font size. Shrink the color bar for a better fit.
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Point Density', fontsize=22)

    # --- 6. Add the y=x Reference Line ---
    if simetric:
        x_mean = x_flat.mean()
        y_mean = y_flat.mean()
        x_std = x_flat.std()
        y_std = y_flat.std()
        min_val = min(x_mean - 5*x_std, y_mean - 5* y_std)
        max_val = max(x_mean + 5*x_std, y_mean + 5* y_std)
        line_vals = np.linspace(min_val, max_val, 100)
        ax.plot(line_vals, line_vals,
                color='gray',
                linestyle='--',
                linewidth=2,
                label='y=x Reference Line')
        # Set the common axis limits for a square frame
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_xlim(0, np.median(x_flat))
        ax.set_ylim(0, np.median(y_flat))

    # --- 7. Set Plot Labels, Title, and Legend with increased font sizes ---
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=12)

    # Increase the font size of the tick labels on both axes and add minor ticks
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Ensure the plot area is square
    
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif', 'Bitstream Vera Serif']

    plt.tight_layout()

    # --- 8. Save or Show the Plot ---
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved successfully to '{save_path}'")
    else:
        plt.show()
        
        
        
def plot_line_in_domain(target, output, save_path="line_quarters", along="z"):
    """
    Plots the fluid prevision and target lines at the center of four domain quarters.
    
    Args:
        target (np.ndarray): The 3D numpy array representing the target fluid data.
        output (np.ndarray): The 3D numpy array representing the predicted fluid data.
        save_path (str): The base path to save the output image.
        along (str): The axis to plot along ('z', 'x', or 'y').
    """
    # Define a modern, professional color palette
    colors = {
        'predicted': '#1f77b4',  # A nice blue
        'actual': '#ff7f0e',     # A nice orange
        'grid': '#cccccc',       # A light grey for the grid
        'background': '#f5f5f5'  # A very light grey background
    }
    
    # Get the dimensions of the domain
    dim_z, dim_y, dim_x = target.shape

    # Define the coordinates for the center of each of the four quarters
    quarter_coordinates = [
        (dim_y // 4, dim_x // 4),
        (dim_y // 4, 3 * dim_x // 4),
        (3 * dim_y // 4, dim_x // 4),
        (3 * dim_y // 4, 3 * dim_x // 4)
    ]
    
    # Define titles for each subplot
    quarter_titles = [
        'Top-Left Quarter',
        'Top-Right Quarter',
        'Bottom-Left Quarter',
        'Bottom-Right Quarter'
    ]

    # Set a professional plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create the figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    
    # Set the background color for the figure
    fig.patch.set_facecolor(colors['background'])
    
    # Add a main title for the entire figure
    fig.suptitle(
        'Fluid Prevision vs. Actual Fluid in Four Domain Quarters',
        fontsize=22,
        fontweight='bold',
        y=0.95  # Adjusts vertical position of the title
    )
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    # Loop through each subplot and plot the data
    for i, ax in enumerate(axes_flat):
        q_y, q_x = quarter_coordinates[i]
        
        # Extract data based on the plotting axis
        if along == "z":
            predicted_data = output[:, q_y, q_x]
            target_data = target[:, q_y, q_x]
            time_steps = np.arange(dim_z)
            ax.set_xlabel(f"{along}-axis (X-dimension)", fontsize=12)
        elif along == "x":
            predicted_data = output[dim_z // 2, q_y, :]
            target_data = target[dim_z // 2, q_y, :]
            time_steps = np.arange(dim_x)
            ax.set_xlabel(f"{along}-axis (X-dimension)", fontsize=12)
        else: # along == "y"
            predicted_data = output[dim_z // 2, :, q_x]
            target_data = target[dim_z // 2, :, q_x]
            time_steps = np.arange(dim_y)
            ax.set_xlabel(f"{along}-axis (X-dimension)", fontsize=12)

        # Plot the target and predicted data
        ax.plot(
            time_steps, target_data, 
            color=colors['actual'], 
            linestyle='--', 
            linewidth=3, 
            label='Target'
        )
        ax.plot(
            time_steps, predicted_data, 
            color=colors['predicted'], 
            linestyle='-', 
            linewidth=3, 
            label='Predicted'
        )
        
        # Set subplot title and labels
        ax.set_title(quarter_titles[i], fontsize=16, pad=10)
        ax.set_ylabel('Fluid Value', fontsize=12)
        
        # Customize the appearance of each subplot
        ax.set_facecolor('white')
        ax.legend(loc='best', frameon=False, fontsize=10)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, color=colors['grid'], alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    # Save the figure with a high DPI for high-quality output
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
