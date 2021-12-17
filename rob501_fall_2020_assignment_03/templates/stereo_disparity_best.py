import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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
    #SUMMARY OF APPROACH
    # My approach here was to try and find as simple a solution as possible by
    # building off my code from part 1. The solution I landed on uses the result
    # from part 1, but adds pre and post filtering via a sobel filter and median 
    # filter, respectively. The base algorithm uses the sum of absolute differences (SAD) approach
    # for the pixel-based matching cost. This simple method just involves calculating the 
    # absolute difference between the corresponding pixel match in the left and right image.
    # Based on this cost function, we can then calculate and store the disparity value
    # in the disparity image. However, this method alone was not getting quite the 
    # desired performance to hit the Erms and Pbad requirements, so I attempted to pre 
    # and post process the disparity image using some scipy filters. I tried with a couple
    # different filters but ended up getting the best performance by pre-processing Il and Ir
    # with a Sobel Filter, and then post processing the disparity image with a median filter. 
    # The Sobel Filter improves the stereo correspondance by calculating gradients in the 
    # image and emphasizing edges in the image. The median filter helps to remove noise in the
    # output disparity image. I also attempted to improve the algorithm performance at the 
    # image edges by using the numpy pad function to extend the edge pixels by half the window size.
 
    #Calculate Parameters and Settings, define window size
    size = Il.shape
    Id = np.zeros(size)  #initialize disparity image with same size as Il
    window=4
    half_window=int(window/2)
    
    #Extend pixels at edges by half of window size for both right and left image
    Ilp = np.pad(Il, window, 'edge' )
    Irp = np.pad(Ir, window, 'edge' )
    
    #Pre-process right and left images by applying a sobel filter to the padded images
    Ilp=sobel(Ilp,axis=0)
    Irp=sobel(Irp,axis=0)

    #Get coordinates of the bounding box
    c1,c2 = bbox[:,0]   
    c3,c4 = bbox[:,1]
    
    #MAIN LOOP

    #diff = 10000000000
    a=c3+half_window
    b=c4+half_window
    disparity = 0
    for xl in range(c1,a,half_window):
        for yl in range(c2,b,half_window):
            diff = 10000000000   #initialize the maximum difference to a large number
            #disparity = 0
            
            #find match in right image
            match = Ilp[yl-half_window:yl+half_window, xl-half_window:xl+half_window]
            flat_match = match.flatten()
          
            #Now, we only need to iterate over x in right image due to fronto-parallel configuration
            for xr in range(xl,xl-maxd-1,-1): 
                #error handling
                check = xr-half_window
                check2 = xr+half_window
                if check<0 or check2 > size[1]:
                    continue
                right = Irp[yl-half_window:yl+half_window,xr-half_window:xr+half_window]
                flat_right = right.flatten()
                
                #apply sad
                arr = np.abs(flat_match - flat_right)
                SAD = np.sum(arr) #sum of absolute differences
                if SAD < diff:
                    diff = SAD
                    disparity = xl-xr
            #print(disparity)
            #Id[yl:yl+window,xl:xl+window] = disparity
            Id[yl-half_window:yl+half_window,xl-half_window:xl+half_window] = disparity
            #Id=median_filter(Id,size=10)
            
    
    
    Id=median_filter(Id,size=10)
    #Id = maximum_filter(Id,10)

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id