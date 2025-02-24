import numpy as np
import pyvista as pv
import panel as pn
import tempfile
import os
from pathlib import Path
from src.utils.utils import VAR


def create_mesh(mesh_data, faces, var_index, var, offset=np.array([0, 0, 0]), n=None):
    points = mesh_data[:, :3] + offset

    cells = faces[:, 1:].transpose(1, 0) if faces[:, 1:].shape[0] == 4 else faces[:, 1:]
    stress = mesh_data[:, var_index]
    if n is not None:
        n_act = np.argwhere(n == 1)
        points = points[:n_act.min()]
        stress = stress[:n_act.min()]
        cells = cells[np.unique(np.argwhere(cells < n_act.min())[:, 0])]

    num_cells = cells.shape[0]
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
    cells_with_sizes = np.hstack([np.full((num_cells, 1), 4), cells]).flatten()
    grid = pv.UnstructuredGrid(cells_with_sizes, cell_types, points)
    #grid = pv.PolyData(points)
    grid[f"{var}"] = stress
    return grid

def make_pyvista_video_combined(predictions, targets, faces, var='S.Mises', max_steps=400,
                       path=None, output_file="output_video", n=None):
    FACTOR = 5
    var_index = -1 if var=='S.Mises' else VAR.index(var)
    # Calculate the minimum and maximum stress values across all steps
    all_stress_values = np.concatenate([mesh_data[:, var_index] for mesh_data in targets])
    min_stress = all_stress_values.min()
    max_stress = all_stress_values.max()
    # Create the list of meshes
    meshes_pred = [create_mesh(predictions[i], faces, var_index, var, n=n) for i in range(0, max_steps, FACTOR)]
    meshes_target = [create_mesh(targets[i], faces, var_index, var,  n=n) for i in range(0, max_steps, FACTOR)]
    # Initialize Plotter
    pl = pv.Plotter(window_size=(2048, 1024), shape=(1, 2), off_screen=True)
    pl.link_views()
    offset = np.array([0, 0, 0])
    #pl.camera_position = [(1., 0.9, 0.8),# (a,b,c) b distance to object, c angle, termino 1 up 0 down
    #(0.25, 0.125, 0.),# (a, b, c) c don touch
    #(0.0, 0.0, 1)]
    # Function to add text for each step
    def add_step_text(step, type='Prediction'):
        return pl.add_text(f"{type} Step: {int(step * FACTOR)}", position='upper_edge', font_size=15, color='black')

    # Generate the frames
    with tempfile.TemporaryDirectory() as tmpdirname:
        video_path = os.path.join(tmpdirname, "temp_video.mp4")
        pl.open_movie(video_path, framerate=12)

        for i, mesh_pred in enumerate(meshes_pred):
            pl.clear()
            # First Subplot with prediction
            pl.subplot(0, 0)
            pl.add_mesh(mesh_pred, scalars=f"{var}", show_edges=True,
                        clim=[min_stress, max_stress], show_scalar_bar=True)  # Set common color scale

            add_step_text(i, 'Prediction')
            # Second Subplot with ground truth
            pl.subplot(0, 1)
            pl.add_mesh(meshes_target[i], scalars=f"{var}", show_edges=True,
                        clim=[min_stress, max_stress], show_scalar_bar=False)  # Set common color scale
            add_step_text(i, 'Ground truth')

            # Write frames
            pl.write_frame()

        pl.close()
        output_file = output_file + '.mp4'
        # Move the temp video file to the final output file
        if path is None:
            os.rename(video_path, output_file)
        else:
            path = Path(path)
            path.mkdir(exist_ok=True)
            os.rename(video_path, str(path / output_file))
            output_file = str(path / output_file)

    print(f"Video saved as {output_file}")

    return output_file

def make_pyvista_interactive_plot(predictions, targets, faces):
    plotter = pv.Plotter()

    camera_position = [(1.0, 1.0, 1.0),
                       (0.0, 1.0, 0.0),
                       (0.0, 0.0, 1.0)]
    plotter.camera_position = camera_position

    # Create a list to store the mesh actors
    mesh_actors = []
    # Calculate the minimum and maximum stress values across all steps
    all_stress_values = [stress for mesh_data in predictions for stress in mesh_data[:, -1]]
    min_stress = min(all_stress_values)
    max_stress = max(all_stress_values)

    def create_mesh(mesh_data, faces, offset=np.array([0, 0, 0])):
        points = mesh_data[:, :3] + offset
        cells = faces[:, 1:].transpose(1, 0)
        stress = mesh_data[:, -1]
        num_cells = cells.shape[0]
        cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells_with_sizes = np.hstack([np.full((num_cells, 1), 4), cells]).flatten()
        grid = pv.UnstructuredGrid(cells_with_sizes, cell_types, points)
        grid["stress"] = stress
        return grid

    # Create a function to update the mesh
    def update_mesh(slider_value):
        global camera_position

        camera_position = plotter.camera_position

        step = int(round(slider_value))

        # Clear all previous actors
        for actor in mesh_actors:
            plotter.remove_actor(actor)
        mesh_actors.clear()

        # Create and add new mesh for predictions
        grid_pred = create_mesh(predictions[step], faces)
        mesh_actor_pred = plotter.add_mesh(grid_pred, show_edges=True, clim=[min_stress, max_stress],
                                           show_scalar_bar=False)
        mesh_actors.append(mesh_actor_pred)

        # Add title for predictions
        plotter.add_text(f'Prediction', position='upper_left', font_size=20, color='black')

        # Create and add new mesh for targets
        grid_targets = create_mesh(targets[step], faces, offset=np.array([0.7, 0, 0]))
        mesh_actor_targets = plotter.add_mesh(grid_targets, show_edges=True, clim=[min_stress, max_stress],
                                              show_scalar_bar=False)
        mesh_actors.append(mesh_actor_targets)

        plotter.add_text(f'Target', position='upper_right', font_size=20, color='black')

        # Create scalar bar
        scalar_bar = plotter.add_scalar_bar(title="Stress", n_labels=4, vertical=True, title_font_size=20,
                                            position_x=0,
                                            width=0.05, height=0.8)
        plotter.camera_position = camera_position
        plotter.view_xy()

    # Add the initial mesh
    update_mesh(0)

    slider_widget = plotter.add_slider_widget(
        callback=update_mesh,  # Ensure the slider value is an integer
        rng=[0, 50],  # Set range according to the length of predictions
        value=0,  # Initial value
        title='Simulation Step',
        title_height=0.02,  # Title height for better visibility
        title_opacity=0.6,  # Title opacity
        style='modern',  # Slider style
        pointa=(0.25, 0.07),  # Start point of the slider
        pointb=(0.75, 0.07),  # End point of the slider
    )

    # Show the interactive plot
    plotter.show(interactive=True)


def make_pyvista_video(predictions, targets, faces, var_index=-1, max_steps=400, path=None, output_file="output_video", with_points=False):

    FACTOR = 5

    var = VAR[var_index]

    # Calculate the minimum and maximum stress values across all steps
    all_stress_values = np.concatenate([mesh_data[:, var_index] for mesh_data in targets])
    min_stress = all_stress_values.min()
    max_stress = all_stress_values.max()

    # Function to create a mesh from the given data
    def create_mesh(mesh_data, faces, var_index, offset=np.array([0, 0, 0])):
        points = mesh_data[:, :3] + offset
        cells = faces[:, 1:].transpose(1, 0) if faces[:, 1:].shape[0] == 4 else faces[:, 1:]
        stress = mesh_data[:, var_index]
        num_cells = cells.shape[0]
        cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells_with_sizes = np.hstack([np.full((num_cells, 1), 4), cells]).flatten()
        grid = pv.UnstructuredGrid(cells_with_sizes, cell_types, points)
        #grid = pv.PolyData(points)
        grid[f"{var}"] = stress
        return grid

    # Create the list of meshes
    meshes_pred = [create_mesh(predictions[i], faces, var_index) for i in range(0, max_steps, FACTOR)]
    meshes_target = [create_mesh(targets[i], faces, var_index) for i in range(0, max_steps, FACTOR)]

    # Initialize Plotter
    pl = pv.Plotter(window_size=(2048, 1024), shape=(1, 2), off_screen=True)
    pl.link_views()

    # Function to add text for each step
    def add_step_text(step, type='Prediction'):
        return pl.add_text(f"{type} Step: {int(step * FACTOR)}", position='upper_edge', font_size=15, color='black')

    # Generate the frames
    with tempfile.TemporaryDirectory() as tmpdirname:
        video_path = os.path.join(tmpdirname, "temp_video.mp4")
        pl.open_movie(video_path, framerate=12)

        for i, mesh_pred in enumerate(meshes_pred):
            pl.clear()
            # First Subplot with prediction
            pl.subplot(0, 0)
            pl.add_mesh(mesh_pred, scalars=f"{var}", show_edges=True,
                        clim=[min_stress, max_stress], show_scalar_bar=False)  # Set common color scale

            add_step_text(i, 'Prediction')
            # Second Subplot with ground truth
            pl.subplot(0, 1)
            pl.add_mesh(meshes_target[i], scalars=f"{var}", show_edges=True,
                        clim=[min_stress, max_stress], show_scalar_bar=False)  # Set common color scale
            add_step_text(i, 'Ground truth')

            # Write frames
            pl.write_frame()

        pl.close()
        output_file = output_file + '.mp4'
        # Move the temp video file to the final output file
        if path is None:
            os.rename(video_path, output_file)
        else:
            path = Path(path)
            path.mkdir(exist_ok=True)
            os.rename(video_path, str(path / output_file))
            output_file = str(path / output_file)

    print(f"Video saved as {output_file}")

    return output_file


def make_pyvista_video_single(predictions, faces, var_index=-1, max_steps=400, path=None, output_file="output_video", with_points=False):

    FACTOR = 10
    var = VAR[var_index]

    # Calculate the minimum and maximum stress values across all steps
    all_stress_values = np.concatenate([mesh_data[:, var_index] for mesh_data in predictions])
    min_stress = all_stress_values.min()
    max_stress = all_stress_values.max()

    # Function to create a mesh from the given data
    def create_mesh(mesh_data, faces, var_index, offset=np.array([0, 0, 0])):
        points = mesh_data[:, :3] + offset
        cells = faces[:, 1:].transpose(1, 0) if faces[:, 1:].shape[0] == 4 else faces[:, 1:]
        stress = mesh_data[:, var_index]
        num_cells = cells.shape[0]
        cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells_with_sizes = np.hstack([np.full((num_cells, 1), 4), cells]).flatten()
        grid = pv.UnstructuredGrid(cells_with_sizes, cell_types, points)
        #grid = pv.PolyData(points)
        grid[f"{var}"] = stress
        return grid

    # Create the list of meshes
    meshes_pred = [create_mesh(predictions[i], faces, var_index) for i in range(0, max_steps, FACTOR)]

    # Initialize Plotter
    pl = pv.Plotter(window_size=(2048, 1024), off_screen=True)
    pl.link_views()

    # Function to add text for each step
    def add_step_text(step, type='Prediction'):
        return pl.add_text(f"{type} Step: {int(step * FACTOR)}", position='upper_edge', font_size=15, color='black')

    # Generate the frames
    with tempfile.TemporaryDirectory() as tmpdirname:
        video_path = os.path.join(tmpdirname, "temp_video.mp4")
        pl.open_movie(video_path, framerate=12)

        for i, mesh_pred in enumerate(meshes_pred):
            pl.clear()
            # First Subplot with prediction
            pl.add_mesh(mesh_pred, show_edges=True, clim=[min_stress, max_stress], show_scalar_bar=True,
                        scalar_bar_args={"title": f"{var}", "vertical": True, "position_x": 0.9, "position_y": 0.15, "height": 0.7, "width": 0.05}
)  # Set common color scale
            add_step_text(i, '')
            # Write frames
            pl.write_frame()

        pl.close()
        output_file = output_file + '.mp4'
        # Move the temp video file to the final output file
        if path is None:
            os.rename(video_path, output_file)
        else:
            path = Path(path)
            path.mkdir(exist_ok=True)
            os.rename(video_path, str(path / output_file))
            output_file = str(path / output_file)

    print(f"Video saved as {output_file}")

    return output_file


def make_pyvista_interactive_panel(predictions, targets, faces, max_steps=400):
    pn.extension("vtk", sizing_mode="stretch_width", template='fast')

    # Calculate the minimum and maximum stress values across all steps
    all_stress_values = [stress for mesh_data in predictions for stress in mesh_data[:, -1]]
    min_stress = min(all_stress_values)
    max_stress = max(all_stress_values)

    # Function to create a mesh from the given data
    def create_mesh(mesh_data, faces, offset=np.array([0, 0, 0])):
        points = mesh_data[:, :3] + offset
        cells = faces[:, 1:].transpose(1, 0)
        stress = mesh_data[:, -1]
        num_cells = cells.shape[0]
        cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells_with_sizes = np.hstack([np.full((num_cells, 1), 4), cells]).flatten()
        grid = pv.UnstructuredGrid(cells_with_sizes, cell_types, points)
        grid["stress"] = stress
        return grid

    meshes = [create_mesh(predictions[i], faces) for i in range(0, max_steps, 10)]

    # Initialize Plotter
    pl = pv.Plotter()
    # Add the first mesh to the plotter
    pl.add_mesh(meshes[0], show_edges=True, clim=[min_stress, max_stress], show_scalar_bar=False)
    pl.view_xy()
    # Clone the panel to animate only this panel
    pan_clone = pn.panel(pl.ren_win, orientation_widget=True, width=1000, height=1000)
    pan_clone.unlink_camera()  # We don't want to share the camera with the previous panel
    # Create a player widget for animation control with a faster interval
    player = pn.widgets.DiscretePlayer(name='Player', options=[i for i in range(len(meshes))], value=0, interval=50)

    # Define animate function
    def animate(value):
        # Clear the previous mesh
        pl.clear()
        # Add the next mesh to the plotter
        pl.add_mesh(meshes[value.new], show_edges=True, clim=[min_stress, max_stress], show_scalar_bar=False)
        # Update the step text
        pl.add_text(f"Step: {value.new}", position='upper_left', font_size=20, color='black')
        # Synchronize changes to the panel
        pan_clone.synchronize()

    # Define callback for player widget
    player.param.watch(animate, 'value')
    # Serve the interactive plot
    pn.FloatPanel(pan_clone, player).show()


def plot_mesh_pyvista(data, face):

    # Extract node positions and cell connectivity
    points = data[:, :3]
    stress = data[:, -1]
    cells = face[:, 1:].transpose(1, 0)

    # Convert cells to pyVista format
    # For each cell, prepend the number of points in the cell
    num_cells = cells.shape[0]
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)  # Assuming quad cells

    # Prepend the number of points per cell
    cells_with_sizes = np.hstack([np.full((num_cells, 1), 4), cells]).flatten()

    # Create the pyVista UnstructuredGrid
    grid = pv.UnstructuredGrid(cells_with_sizes, cell_types, points)

    # Optionally, add stress data as a cell array
    grid["stress"] = stress

    # Plot the mesh with stress values
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="stress", show_edges=True)

    return plotter

