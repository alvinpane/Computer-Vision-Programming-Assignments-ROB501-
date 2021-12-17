import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---
    
    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    
    #obtain the surrounding 4 pixels
    x_1 = int(pt[0]) - 1
    x_2 = int(pt[0]) + 1
    y_1 = int(pt[1]) - 1
    y_2 = int(pt[1]) + 1

    # we will use the polynomial fit method to determine the bilinear interpolation
    A = np.array([[1, x_1, y_1, x_1 * y_1],[1, x_1, y_2, x_1 * y_2],[1, x_2, y_1, x_2 * y_1],[1, x_2, y_2, x_2 * y_2]])
    A_inv = inv(A)
    
    Q = np.array([[I[y_1, x_1]],[I[y_2, x_1]],[I[y_1, x_2]], [I[y_2, x_2]]])

    x = A_inv @ Q

    
    b = int(x[0] + x[1] * pt[0] + x[2] * pt[1] + x[3] * pt[0]* pt[1])    #compute polynomial expression for intensity
    b = round(b)    #round b

    #------------------

    return b
