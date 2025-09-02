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

def plot_distributions(dict_data, normalize=False, title="Distribution Comparison", save_path="plot.svg"):
    """
    Compare distributions of multiple arrays from a dictionary.
    
    Args:
        dict_data (dict): Dictionary of {name: array} pairs
        normalize (bool): Whether to normalize densities independently
        title (str): Plot title
    """
    plt.figure(figsize=(12, 7))
    ax = plt.gca()  # Get the current axes
    base_colors = list(mcolors.BASE_COLORS.keys())
    
    for i, (name, array) in enumerate(dict_data.items()):
        array = array.flatten()
        mean = np.mean(array)
        std = np.std(array)
        
        # Calculate the mode
        try:
            moda = mode(array)[0]
        except TypeError:
            # Handle cases where the array might be empty or other issues
            moda = np.nan
            
        sns.kdeplot(
            array, 
            label=f'{name} (μ={mean:.4f}, σ={std:.4f}, moda={moda:.4f})',
            color=base_colors[i % len(base_colors)],
            fill=True, 
            common_norm=not normalize
        )
        

    # Labels and title with increased font size
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(title, fontsize=16)

    # Increase tick font sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Legend and grid
    plt.legend(fontsize=12)
    
    # Enable minor ticks and set their frequency
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path + ".png", bbox_inches='tight')
    plt.show()


def plot_joint_frequency(distance_transform, velocity, cmap='plasma',
                         title="Joint Frequency", xaxis="Distance", yaxis="Velocity", save_path="./plot.svg"):
    # Flatten input arrays
    dist_flat = distance_transform.flatten()
    vel_flat = velocity.flatten()

    if len(dist_flat) != len(vel_flat):
        raise ValueError("Input arrays must have the same number of elements")

    # Define number of bins
    bins = int(np.ceil(np.log2(len(vel_flat)) + 1))
    
    # Define fixed bin edges (uniform)
    xedges = np.linspace(dist_flat.min(), dist_flat.max(), bins + 1)
    yedges = np.linspace(vel_flat.min(), vel_flat.max(), bins + 1)
    
    # Compute histogram
    hist, _, _ = np.histogram2d(dist_flat, vel_flat, bins=[xedges, yedges])
    
    # Normalize the histogram to represent percentages [0-100%]
    total_occurrences = len(dist_flat)
    hist_percentage = (hist / total_occurrences) * 100.0
    
    # Handle zero values for visualization
    hist_percentage = np.where(hist_percentage == 0, 1e-10, hist_percentage)

    # Create a single plot with a color bar
    fig, ax = plt.subplots(figsize=(10, 8))

    # Main heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    mesh = ax.imshow(hist_percentage.T, origin='lower', extent=extent,
                     aspect='auto', cmap=cmap)
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, pad=0.05)
    cbar.set_label('Frequency (%)', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Axes labels and title
    ax.set_xlabel(xaxis, labelpad=10, fontsize=14)
    ax.set_ylabel(yaxis, labelpad=10, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Save and show
    plt.savefig(save_path + ".png", bbox_inches='tight')
    plt.show()

