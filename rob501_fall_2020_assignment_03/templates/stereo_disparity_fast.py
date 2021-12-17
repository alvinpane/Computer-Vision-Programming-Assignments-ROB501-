import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Code goes here...
    size = Il.shape
    Id = np.zeros(size)  #initialize disparity image with same size as Il
    
    c1,c2 = bbox[:,0]   #coords of the bounding box, use to find overlap
    c3,c4 = bbox[:,1]
    overlap = Il[c2:c4,c1:c3]
    
    #MAIN LOOP
    #define a window size and initialize max diff to a large number
    window=8
    half_window=int(window/2)
    #diff = 10000000000
    a=c3-window
    b=c4-window
    disparity = 0
    for xl in range(c1,a,window):
        for yl in range(c2,b,window):
            diff = 10000000000
            #disparity = 0
            match = Il[yl-half_window:yl+half_window, xl-half_window:xl+half_window]
            flat_match = match.flatten()
            #find match in right image
            for xr in range(xl,xl-maxd-1,-1): #only concerned about x in right image due to fronto-parallel config
                #handling
                check = xr-half_window
                check2 = xr+half_window
                if check<0 or check2 > size[1]:
                    continue
                right = Ir[yl-half_window:yl+half_window,xr-half_window:xr+half_window]
                flat_right = right.flatten()
                #apply ssd
                arr = np.square(flat_match - flat_right)
                SSD = np.sum(arr) #sum of squared differences
                if SSD < diff:
                    diff = SSD
                    disparity = xl-xr
            #print(disparity)
            Id[yl:yl+window,xl:xl+window] = disparity
    
    

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id