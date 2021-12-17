import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---
    if I.dtype != np.uint8:   # Verify I is grayscale.
        raise ValueError('Incorrect image format!')
    
    shaped = I.shape
    flat_image = I.flatten() #flatten image
    H, bins = np.histogram(flat_image, bins = 255, range = (0,255))
    C = H.cumsum() #cumulative density function
    last=C[-1]
    C = 255 * C / last #normalize CDF
    # Linear interpolation of CDF to find new pixel values
    J = np.interp(flat_image, bins[:-1], C)
            
    J = J.reshape(shaped)
    #-------------------

    return J
