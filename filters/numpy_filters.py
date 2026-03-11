"""
NumPy Image Processing Filters
================================
Implementation of Gaussian, Sobel, and Median filters
using NumPy vectorized operations for optimized performance.

Authors: Raúl Cetina, Christian Carreño, Christopher Quiñones
"""

import numpy as np


def gaussian_filter(image):
    """
    Apply a 3x3 Gaussian blur filter using NumPy vectorized operations.

    The Gaussian kernel used is:
        G = (1/16) * [[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array representing the grayscale image (dtype uint8).

    Returns
    -------
    np.ndarray
        The blurred image as a 2D NumPy array (dtype uint8).
    """
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16.0

    rows, cols = image.shape
    output = np.zeros_like(image, dtype=np.float64)

    # Vectorized 2D convolution using array slicing
    for ki in range(3):
        for kj in range(3):
            output[1:rows-1, 1:cols-1] += (
                image[ki:rows-2+ki, kj:cols-2+kj].astype(np.float64) * kernel[ki, kj]
            )

    # Copy border pixels
    output[0, :] = image[0, :]
    output[-1, :] = image[-1, :]
    output[:, 0] = image[:, 0]
    output[:, -1] = image[:, -1]

    return np.clip(output, 0, 255).astype(np.uint8)


def sobel_filter(image):
    """
    Apply the Sobel edge detection filter using NumPy vectorized operations.

    Uses the standard Sobel kernels:
        Sx = [[-1, 0, 1],    Sy = [[-1, -2, -1],
              [-2, 0, 2],          [ 0,  0,  0],
              [-1, 0, 1]]          [ 1,  2,  1]]

    Gradient magnitude: G = sqrt(Sx^2 + Sy^2)

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array representing the grayscale image.

    Returns
    -------
    np.ndarray
        The edge-detected image as a 2D NumPy array (dtype uint8).
    """
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    rows, cols = image.shape
    gx = np.zeros((rows, cols), dtype=np.float64)
    gy = np.zeros((rows, cols), dtype=np.float64)

    # Vectorized convolution for both kernels
    for ki in range(3):
        for kj in range(3):
            region = image[ki:rows-2+ki, kj:cols-2+kj].astype(np.float64)
            gx[1:rows-1, 1:cols-1] += region * kernel_x[ki, kj]
            gy[1:rows-1, 1:cols-1] += region * kernel_y[ki, kj]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def median_filter(image):
    """
    Apply a 3x3 median filter using NumPy operations.

    Collects all 9 neighbors into a stacked array and computes
    the median along the stack axis for vectorized performance.

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array representing the grayscale image.

    Returns
    -------
    np.ndarray
        The filtered image as a 2D NumPy array (dtype uint8).
    """
    rows, cols = image.shape
    output = np.copy(image)

    # Stack all 9 neighbors for the interior region
    neighbors = np.zeros((9, rows - 2, cols - 2), dtype=np.uint8)
    idx = 0
    for ki in range(3):
        for kj in range(3):
            neighbors[idx] = image[ki:rows-2+ki, kj:cols-2+kj]
            idx += 1

    # Compute median along the stack axis
    output[1:rows-1, 1:cols-1] = np.median(neighbors, axis=0).astype(np.uint8)

    return output
