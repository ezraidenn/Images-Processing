# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython Image Processing Filters
=================================
Implementation of Gaussian, Sobel, and Median filters
using Cython typed memoryviews for C-level performance.

Authors: Raúl Cetina, Daniel Gómez, Christopher Quiñones
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


def gaussian_filter(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    """
    Apply a 3x3 Gaussian blur filter using Cython typed operations.

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array (uint8) representing the grayscale image.

    Returns
    -------
    np.ndarray
        The blurred image as a 2D NumPy array (uint8).
    """
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.zeros((rows, cols), dtype=np.uint8)
    cdef int i, j
    cdef int pixel_sum
    cdef int kernel[3][3]

    # Gaussian kernel (weights, will divide by 16)
    kernel[0][0] = 1; kernel[0][1] = 2; kernel[0][2] = 1
    kernel[1][0] = 2; kernel[1][1] = 4; kernel[1][2] = 2
    kernel[2][0] = 1; kernel[2][1] = 2; kernel[2][2] = 1

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            pixel_sum = (
                image[i-1, j-1] * kernel[0][0] +
                image[i-1, j  ] * kernel[0][1] +
                image[i-1, j+1] * kernel[0][2] +
                image[i  , j-1] * kernel[1][0] +
                image[i  , j  ] * kernel[1][1] +
                image[i  , j+1] * kernel[1][2] +
                image[i+1, j-1] * kernel[2][0] +
                image[i+1, j  ] * kernel[2][1] +
                image[i+1, j+1] * kernel[2][2]
            )
            pixel_sum = pixel_sum // 16
            if pixel_sum > 255:
                pixel_sum = 255
            elif pixel_sum < 0:
                pixel_sum = 0
            output[i, j] = <cnp.uint8_t>pixel_sum

    # Copy border pixels
    for i in range(rows):
        output[i, 0] = image[i, 0]
        output[i, cols - 1] = image[i, cols - 1]
    for j in range(cols):
        output[0, j] = image[0, j]
        output[rows - 1, j] = image[rows - 1, j]

    return output


def sobel_filter(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    """
    Apply the Sobel edge detection filter using Cython typed operations.

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array (uint8) representing the grayscale image.

    Returns
    -------
    np.ndarray
        The edge-detected image as a 2D NumPy array (uint8).
    """
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.zeros((rows, cols), dtype=np.uint8)
    cdef int i, j
    cdef double gx, gy, magnitude

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Sobel X kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
            gx = (
                -1.0 * image[i-1, j-1] + 1.0 * image[i-1, j+1] +
                -2.0 * image[i  , j-1] + 2.0 * image[i  , j+1] +
                -1.0 * image[i+1, j-1] + 1.0 * image[i+1, j+1]
            )
            # Sobel Y kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
            gy = (
                -1.0 * image[i-1, j-1] - 2.0 * image[i-1, j] - 1.0 * image[i-1, j+1] +
                 1.0 * image[i+1, j-1] + 2.0 * image[i+1, j] + 1.0 * image[i+1, j+1]
            )
            magnitude = sqrt(gx * gx + gy * gy)
            if magnitude > 255.0:
                magnitude = 255.0
            output[i, j] = <cnp.uint8_t>magnitude

    return output


def median_filter(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    """
    Apply a 3x3 median filter using Cython typed operations.

    Uses an optimized sorting network for 9 elements.

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array (uint8) representing the grayscale image.

    Returns
    -------
    np.ndarray
        The filtered image as a 2D NumPy array (uint8).
    """
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.zeros((rows, cols), dtype=np.uint8)
    cdef int i, j, k, m
    cdef int neighbors[9]
    cdef int temp

    # Copy border pixels
    for i in range(rows):
        output[i, 0] = image[i, 0]
        output[i, cols - 1] = image[i, cols - 1]
    for j in range(cols):
        output[0, j] = image[0, j]
        output[rows - 1, j] = image[rows - 1, j]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Collect 3x3 neighborhood
            neighbors[0] = image[i-1, j-1]
            neighbors[1] = image[i-1, j  ]
            neighbors[2] = image[i-1, j+1]
            neighbors[3] = image[i  , j-1]
            neighbors[4] = image[i  , j  ]
            neighbors[5] = image[i  , j+1]
            neighbors[6] = image[i+1, j-1]
            neighbors[7] = image[i+1, j  ]
            neighbors[8] = image[i+1, j+1]

            # Simple insertion sort for 9 elements (very efficient for small N)
            for k in range(1, 9):
                temp = neighbors[k]
                m = k - 1
                while m >= 0 and neighbors[m] > temp:
                    neighbors[m + 1] = neighbors[m]
                    m -= 1
                neighbors[m + 1] = temp

            output[i, j] = <cnp.uint8_t>neighbors[4]  # Median

    return output
