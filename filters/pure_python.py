"""
Pure Python Image Processing Filters
=====================================
Implementation of Gaussian, Sobel, and Median filters
using only Python built-in operations (no NumPy).

Authors: Raúl Cetina, Daniel Gómez, Christopher Quiñones
"""

import math


def gaussian_filter(image):
    """
    Apply a 3x3 Gaussian blur filter to a grayscale image.

    The Gaussian kernel used is:
        G = (1/16) * [[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]

    Parameters
    ----------
    image : list of list of int
        2D list representing the grayscale image (pixel values 0-255).

    Returns
    -------
    list of list of int
        The blurred image as a 2D list.
    """
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    kernel_sum = 16  # Sum of all kernel weights

    rows = len(image)
    cols = len(image[0])
    output = [[0] * cols for _ in range(rows)]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            pixel_sum = 0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    pixel_sum += image[i + ki][j + kj] * kernel[ki + 1][kj + 1]
            output[i][j] = min(255, max(0, pixel_sum // kernel_sum))

    # Copy border pixels from original image
    for i in range(rows):
        output[i][0] = image[i][0]
        output[i][cols - 1] = image[i][cols - 1]
    for j in range(cols):
        output[0][j] = image[0][j]
        output[rows - 1][j] = image[rows - 1][j]

    return output


def sobel_filter(image):
    """
    Apply the Sobel edge detection filter to a grayscale image.

    Uses the standard Sobel kernels:
        Sx = [[-1, 0, 1],    Sy = [[-1, -2, -1],
              [-2, 0, 2],          [ 0,  0,  0],
              [-1, 0, 1]]          [ 1,  2,  1]]

    The gradient magnitude is computed as: G = sqrt(Sx^2 + Sy^2)

    Parameters
    ----------
    image : list of list of int
        2D list representing the grayscale image.

    Returns
    -------
    list of list of int
        The edge-detected image as a 2D list.
    """
    kernel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    kernel_y = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]

    rows = len(image)
    cols = len(image[0])
    output = [[0] * cols for _ in range(rows)]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = 0
            gy = 0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    pixel = image[i + ki][j + kj]
                    gx += pixel * kernel_x[ki + 1][kj + 1]
                    gy += pixel * kernel_y[ki + 1][kj + 1]
            magnitude = int(math.sqrt(gx * gx + gy * gy))
            output[i][j] = min(255, magnitude)

    return output


def median_filter(image):
    """
    Apply a 3x3 median filter to a grayscale image.

    Replaces each pixel with the median value of its 3x3 neighborhood.
    Effective for removing salt-and-pepper noise while preserving edges.

    Parameters
    ----------
    image : list of list of int
        2D list representing the grayscale image.

    Returns
    -------
    list of list of int
        The filtered image as a 2D list.
    """
    rows = len(image)
    cols = len(image[0])
    output = [[0] * cols for _ in range(rows)]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = []
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    neighbors.append(image[i + ki][j + kj])
            neighbors.sort()
            output[i][j] = neighbors[4]  # Median of 9 elements

    # Copy border pixels
    for i in range(rows):
        output[i][0] = image[i][0]
        output[i][cols - 1] = image[i][cols - 1]
    for j in range(cols):
        output[0][j] = image[0][j]
        output[rows - 1][j] = image[rows - 1][j]

    return output
