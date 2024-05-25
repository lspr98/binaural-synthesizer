import numpy as np


def gaussian(x, sig: float):
    """
    
        Evaluate the zero-centered gaussian kernel with bandwidth sig at x

        Parameters
        ----------
        x:  (N, ) np.array
            points at which to evaluate the gaussian kernel

        sig: float
            Bandwidth of the gaussian kernel

        Returns
        -------
        y:  (N, ) np.array
            Values of the gaussian kernel for each x
    
    """

    return np.exp(-np.abs(x)**2/(2*sig**2))


def get_polar_coords(x, y):
    """

        Converts cartesian coordiantes to polar coordinates

        Parameters
        ----------
        x: float
            distance in first dimension

        y: float
            distance in second dimension

        Returns
        -------
        dist: float
            Distance in meters

        azi_deg: int
            Angle in degrees
    
    """

    # Get distance to origin
    dist = np.linalg.norm([x, y])
    # Get azimuth angle
    azi = np.arctan2(x, y)
    azi_deg = int(np.rad2deg(azi))
    # arctan2 returns values in [-pi pi], so map it to [0 2pi]
    if azi_deg < 0:
        azi_deg += 360

    return dist, azi_deg


def clip_angle_ccw(angle):
    """

        Clips an angle given in degrees to the range [0 359] and converts it from 
        clockwise convention (used by this repository) to counter-clockwise convention (used by the dataset)

        Parameters
        ----------
        angle: int
            Angle in degrees

        Returns
        -------
        angle: int
            Corresponding counter clockwise angle in degrees in range [0 359]
    
    """
    return (360 - (angle % 360)) % 360


def get_polar_coords_ccw(x, y):
    """

        Converts cartesian coordiantes to polar coordinates in counter-clockwise convention

        Parameters
        ----------
        x: float
            distance in first dimension

        y: float
            distance in second dimension

        Returns
        -------
        dist: float
            Distance in meters

        azi_deg: int
            Counter clockwise angle in degrees in range [0 359]
    
    """

    d, theta = get_polar_coords(x, y)
    return d, clip_angle_ccw(theta)