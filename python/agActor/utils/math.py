"""
Mathematical utility functions for the agActor package.
"""

from typing import Tuple

import numpy as np


def semi_axes(xy: float, x2: float, y2: float) -> Tuple[float, float]:
    """
    Calculate the semi-major and semi-minor axes of an ellipse using eigenvalues.
    
    This function computes the semi-axes of an ellipse from the second moments
    of a 2D distribution by calculating the eigenvalues of the covariance matrix.
    
    Parameters
    ----------
    xy : float
        Mixed central moment (covariance) of the distribution.
    x2 : float
        Second central moment in x (variance in x).
    y2 : float
        Second central moment in y (variance in y).
    
    Returns
    -------
    Tuple[float, float]
        A tuple containing the semi-major and semi-minor axes (a, b).
        
    Notes
    -----
    The covariance matrix is formed as:
    [x2  xy]
    [xy  y2]
    
    The eigenvalues of this matrix are used to calculate the semi-axes.
    """
    covariance_matrix = np.array([[x2, xy],
                                  [xy, y2]])

    # eigh returns (eigenvalues, eigenvectors)
    # Eigenvalues are guaranteed to be real. We explicitly sort them in ascending order
    eigenvalues = np.sort(np.linalg.eigh(covariance_matrix)[0])

    # The semi-axes are the square roots of the eigenvalues.
    # Since they are sorted, eigenvalues[1] will be the larger one (semi-major)
    # and eigenvalues[0] will be the smaller one (semi-minor).
    a = np.sqrt(eigenvalues[1]) # Semi-major axis
    b = np.sqrt(eigenvalues[0]) # Semi-minor axis

    return a, b
