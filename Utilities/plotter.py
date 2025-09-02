import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as py
import gc
import os
import torch
import seaborn as sns
from scipy.stats import skew


def Plot_Domain(values, filename, remove_value=[]):
    """
    Plot a 3D domain from a 3D NumPy array and ghost cells with a specific value.

    Parameters:
        values (np.ndarray): 3D NumPy array of cell values.
        filename (str): Name of the output file (without extension).
        remove_value (float/int): Value to mark as ghost cells.
    """
    # Criar grid estruturado (ImageData)
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1  # Dimensões como pontos
    grid.origin = (0, 0, 0)  # Origem da grade
    grid.spacing = (1, 1, 1)  # Espaçamento uniforme

    # Atribuir valores aos dados das células
    grid.cell_data["values"] = values.flatten(order="F")  # Nome do atributo: "values"

    # Remove celulas indesejadas no plot
    mesh = grid.cast_to_unstructured_grid()
    if remove_value:
        for removed_value in remove_value:
            ghosts = np.argwhere(mesh["values"] == removed_value)
            mesh.remove_cells(ghosts.flatten(), inplace=True)

    # Plot the grid with ghosted cells hidden
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)  # Full HD resolution

    # Adiciona celulas
    plotter.add_mesh(
        mesh,
        cmap="viridis",
        show_edges=False,
        lighting=True,
        smooth_shading=True,
        split_sharp_edges=True,
        scalar_bar_args={
            "title": "Range",          # Title of the color bar
            "vertical": True,          # Make the color bar vertical
            "title_font_size": 20,
            "label_font_size": 16,
            "position_x": 0.85,        # Position of the color bar (X-axis)
            "position_y": 0.05,        # Position of the color bar (Y-axis)
            "height": 0.9,             # Height of the color bar
            "width": 0.05              # Width of the color bar
        }
    )

    # Adiciona linhas nas arestas
    edges = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=False,
        feature_angle=30,
        manifold_edges=False,
    )
    plotter.add_mesh(edges,
                     color='k',
                     line_width=5,
                     show_scalar_bar=False)

    # Adiciona indicador de direcao
    plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.9,
        tip_length=0.2,
        ambient=0.5,
        label_size=(0.25, 0.15))

    # Adiciona limites do grafico
    plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        n_xlabels=2,
        n_ylabels=2,
        n_zlabels=2,
        xtitle='x',
        ytitle='y',
        ztitle='z')

    # Cria grafico
    plotter.screenshot(filename + ".png")  # Save as screenshot
    plotter.show()


def plot_3d_planes(array_3d, x_offset=0., y_offset=0., z_offset=0., file_name="3D_planes"):
    # Definir as dimensões do array
    x_dim, y_dim, z_dim = array_3d.shape

    # Calcular o índice central para cada eixo
    x_center = x_dim // 2
    y_center = y_dim // 2
    z_center = z_dim // 2

    # Extrair os planos ao longo dos eixos centrais
    plane_x = array_3d[x_center, :, :]  # Plano no centro de x
    plane_y = array_3d[:, y_center, :]  # Plano no centro de y
    plane_z = array_3d[:, :, z_center]  # Plano no centro de z

    # Coordenadas para os planos com offset
    y_coords, z_coords = np.meshgrid(
        np.arange(y_dim) + y_offset, np.arange(z_dim) + z_offset, indexing="ij")
    x_coords, z_coords_x = np.meshgrid(
        np.arange(x_dim) + x_offset, np.arange(z_dim) + z_offset, indexing="ij")
    x_coords_y, y_coords_y = np.meshgrid(
        np.arange(x_dim) + x_offset, np.arange(y_dim) + y_offset, indexing="ij")

    # Calcular os limites da escala de cores
    # Valor mínimo em todo o array
    cmin = np.nanmin(np.concatenate((plane_x, plane_y, plane_z)))
    # Valor máximo em todo o array
    cmax = np.nanmax(np.concatenate((plane_x, plane_y, plane_z)))

    # Criar a figura com os três planos
    fig = go.Figure()

    # Adicionar plano em x
    fig.add_trace(go.Surface(
        x=np.full_like(plane_x, x_center + x_offset),
        y=y_coords,
        z=z_coords,
        surfacecolor=plane_x,
        colorscale='agsunset',
        cmin=cmin,  # Valor mínimo da escala de cores
        cmax=cmax,  # Valor máximo da escala de cores
        colorbar=dict(title="Valor (Plano X)"),
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
    ))

    # Adicionar plano em y
    fig.add_trace(go.Surface(
        x=x_coords,
        y=np.full_like(plane_y, y_center + y_offset),
        z=z_coords_x,
        surfacecolor=plane_y,
        colorscale='agsunset',
        cmin=cmin,  # Valor mínimo da escala de cores
        cmax=cmax,  # Valor máximo da escala de cores
        colorbar=dict(title="Valor (Plano Y)"),
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
    ))

    # Adicionar plano em z
    fig.add_trace(go.Surface(
        x=x_coords_y,
        y=y_coords_y,
        z=np.full_like(plane_z, z_center + z_offset),
        surfacecolor=plane_z,
        colorscale='agsunset',
        cmin=cmin,  # Valor mínimo da escala de cores
        cmax=cmax,  # Valor máximo da escala de cores
        colorbar=dict(title="Valor (Plano Z)"),
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
    ))

    # Ajustar os limites do gráfico e rótulos
    fig.update_layout(
        title="Cortes 3D Interativos do Domínio com Offset",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(nticks=10, range=[x_offset, x_dim + x_offset]),
            yaxis=dict(nticks=10, range=[y_offset, y_dim + y_offset]),
            zaxis=dict(nticks=10, range=[z_offset, z_dim + z_offset])
        ),
    )

    # Salvar o gráfico em um arquivo HTML para visualização interativa
    fig.write_html(file_name)
    print(f"Gráfico 3D salvo como '{file_name}'")
    

def plot_heatmap(
    array_2d, 
    folder_name="", 
    simulation_name="Example",
    colormap='viridis', 
    xlabel='X-axis', 
    ylabel='Y-axis', 
    title='2D Array Plot', 
    vmin=0,  # Fixed minimum value for the colorscale
    vmax=1,  # Fixed maximum value for the colorscale
    dpi=300, 
    figsize=(8, 6),
    show=False,
    save=True
):
    
    if array_2d.ndim == 3: 
        if array_2d.shape[0] == 1: array_2d = np.squeeze(array_2d, axis=0)  # Check if 3D and first dim is 1
        else: raise Exception("Only 2D or (1, H, L) Arrays allowed")
    """
    Plot a 2D array in HD with a fixed colorscale and save it to a file.

    Parameters:
        array_2d (np.ndarray): The 2D array to plot.
        filename (str): Path to save the output image.
        colormap (str): Colormap to use (default: 'viridis').
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.
        vmin (float): Fixed minimum value for the colorscale.
        vmax (float): Fixed maximum value for the colorscale.
        dpi (int): Resolution of the saved image (default: 300).
        figsize (tuple): Size of the figure in inches (default: (8, 6)).
    """
    plt.figure(figsize=figsize, dpi=dpi)
    if vmin is None or vmax is None: plt.imshow(array_2d, cmap=colormap, interpolation='none')
    else: plt.imshow(array_2d, cmap=colormap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    full_file_path = os.path.join(folder_name, simulation_name+".png")
    if save: plt.savefig(full_file_path, dpi=dpi)
    if show: plt.show()
    plt.close('all')
    gc.collect()
    

# Convert tensors to NumPy and move to CPU if necessary
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor) :
        return tensor.cpu().numpy()
    return tensor
    
def display_image_tensor(tensor, filename):
    img = to_numpy(tensor)

    # (C, H, W) → (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    height, width = img.shape[:2]

    # Set figure size to exactly match the image pixel dimensions
    dpi = 100  # or any dpi you want
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure
    ax.imshow(img)
    ax.axis('off')

    fig.savefig(filename + ".png", dpi=dpi)
    plt.close(fig)
    
    
def display_example(image_tensor, pred_tensor, target_tensor,
                    title="Example", title_image="Input", title_pred="Prediction", title_target="Target"):
    """
    Displays input image, prediction, and target side by side with colorbars.
    Prediction and target share the same color scale for direct comparison.
    """

    print("image_tensor shape:", image_tensor.shape)
    print("pred_tensor shape:", pred_tensor.shape)
    print("target_tensor shape:", target_tensor.shape)

    image_tensor = to_numpy(image_tensor)
    pred_tensor = to_numpy(pred_tensor)
    target_tensor = to_numpy(target_tensor)

    # Determine number of channels for each tensor
    num_channels = {
        "Image": image_tensor.shape[0] if image_tensor.ndim == 3 else 1,
        "Prediction": pred_tensor.shape[0] if pred_tensor.ndim == 3 else 1,
        "Target": target_tensor.shape[0] if target_tensor.ndim == 3 else 1
    }

    # Filter tensors with at least 1 channel
    subplot_tensors = {
        name: tensor for name, tensor in {
            "Image": image_tensor,
            "Prediction": pred_tensor,
            "Target": target_tensor
        }.items() if num_channels[name] > 0
    }

    titles = {
        "Image": title_image,
        "Prediction": title_pred,
        "Target": title_target
    }

    num_rows = len(subplot_tensors)
    max_cols = max(num_channels.values())

    # Compute color ranges
    image_vmin, image_vmax = image_tensor.min(), image_tensor.max()
    shared_min = min(pred_tensor.min(), target_tensor.min())
    shared_max = max(pred_tensor.max(), target_tensor.max())

    # Setup plot
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_cols,
                             figsize=(max_cols * 5, num_rows * 4),
                             squeeze=False, gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

    last_image, last_pred, last_target = None, None, None
    row_idx = 0

    for name, tensor in subplot_tensors.items():
        current_channels = num_channels[name]
        for ch in range(current_channels):
            ax = axes[row_idx, ch]
            img = tensor[ch] if tensor.ndim == 3 else tensor

            if name == "Image":
                im = ax.imshow(img, cmap="gist_gray", vmin=image_vmin, vmax=image_vmax, interpolation='none')
                last_image = im
            elif name == "Prediction":
                im = ax.imshow(img, cmap="plasma", vmin=shared_min, vmax=shared_max, interpolation='none')
                last_pred = im
            elif name == "Target":
                im = ax.imshow(img, cmap="plasma", vmin=shared_min, vmax=shared_max, interpolation='none')
                last_target = im

            ax.set_xticks([])
            ax.set_yticks([])

            if ch == 0:
                ax.set_ylabel(titles[name], fontsize=14, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"Channel {ch}", fontsize=18, fontweight="bold")

        # Hide unused columns
        for ch in range(current_channels, max_cols):
            fig.delaxes(axes[row_idx, ch])

        row_idx += 1

    plt.subplots_adjust(right=0.88)
    
    colorbar_width = 0.015
    for row_idx, (im, label) in enumerate([
        (last_image, "Image Scale"),
        (last_pred, "Prediction/Target Scale"),
        (last_target, None)
    ]):
        if im is None:
            continue
    
        # Get the last used column index for this row
        last_col = num_channels[list(subplot_tensors.keys())[row_idx]] - 1
        pos = axes[row_idx, last_col].get_position()
    
        # Align colorbar after the last plot in the row
        cax = fig.add_axes([
            pos.x1 + 0.01,    # right of the last plot in the row
            pos.y0,
            colorbar_width,
            pos.y1 - pos.y0
        ])
        fig.colorbar(im, cax=cax, orientation="vertical")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.show()
    
