import os
import numpy as np


def create_circle_trajectory(radius: float, period: float, length: float, step: float = 0.1):
    """
    
        Creates a circular trajectory around the listener located at (0, 0). Distances are in meters.

        Parameters
        ----------
        radius: float
            Radius of the circle in meters

        period: float
            Time needed for a full circle in seconds

        length: float
            Total time for trajectory in seconds

        step: float
            Discrete timestep for trajectory in seconds

        Returns
        -------
        (3, N) np.array trajectory, where trajectory[:, i] are the [x, y, t] coordinates for timestep i
    
    """
    
    n_samples = int(length//step)

    t = np.linspace(0, length, n_samples)

    x = radius * np.sin(((2*np.pi)/period)*t)
    y = radius * np.cos(((2*np.pi)/period)*t)

    return np.array([x, y, t])



def export_trajectory(trajectory, output_path: str, force: bool = False):
    """
    
        Writes the trajectory data as a comma-separated textfile, line by line

        Parameters
        ----------
        trajectory: (3, N) np.array
            Trajectory data of N discrete time samples, where trajectory[:, i] are the [x, y, t] coordinates 
            for timestep i. x and y are given in meters, t is given in seconds

        output_path: str
            Output filename and location for trajectory data

        force: bool
            Allow overwriting existing files

        Returns
        -------
        None
    
    """

    assert not os.path.exists(output_path) or force, f"Output path {output_path} already exists."

    assert trajectory.shape[0] == 3, f"Input trajectory is not 3D (found {trajectory.shape[0]} dimensions)"

    n_samples = trajectory.shape[1]

    with open(output_path, "w") as f:
        for i in range(n_samples):
            out_str = ",".join(["{:.2f}".format(coord) for coord in trajectory[:, i]])
            f.write(out_str + "\n")



def import_trajectory(input_path: str):
    """
    
        Loads the trajectory data from a comma-separated textfile

        Parameters
        ----------
        input_path: str
            Input filename and location for trajectory data

        Returns
        -------
        trajectory: (3, N) np.array
            Trajectory data of N discrete time samples, where trajectory[:, i] are the [x, y, t] coordinates 
            for timestep i. x and y are given in meters, t is given in seconds
    
    """

    assert os.path.exists(input_path), f"Input path {input_path} does not exist."

    with open(input_path, "r") as f:
        lines = f.readlines()

    # Sanitized input
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l) > 0]

    arr = []
    for i, line in enumerate(lines):
        coords_str = line.split(",")

        assert len(coords_str) == 3, f"Coordinate in line {i+1} has {len(coords_str)} dimensions, but 3 are required."

        coords = [float(coord) for coord in coords_str]

        arr.append(coords)

    trajectory = np.array(arr).T

    # Sort by increasing time steps
    trajectory = trajectory[:, trajectory[2, :].argsort()]

    # Make sure there is a coordinate given for T=0.0
    assert int(trajectory[2, 0]) == 0, f"Initial time is not 0 (found {trajectory[2, 0]})"

    return trajectory