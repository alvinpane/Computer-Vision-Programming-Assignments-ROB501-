import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    A = np.zeros([8,9])    #create the A matrix, an 8x9 np array
    
    for i,point in enumerate(I1pts.T):   #iterate over the vector of points in Image 1, transposed
        SecondImage = I2pts.T[i]           #find point in second image
        x = point[0]       #x coord 1st pic
        y = point[1]       # y coord 1st pic
        x_2 = SecondImage[0]          #x coord 2nd pic
        y_2 = SecondImage[1]           #y coord 2nd pic
        A_i = np.array([ [-1*x, -1*y, -1, 0 , 0 ,0, x_2*x, x_2*y, x_2],[0 ,0, 0, -1*x, -1*y, -1, y_2*x, y_2*y, y_2],])   
        # from formula in thesis paper ith element of A
        A[i*2 : (i+1)*2,:] = A_i   #finalize A matrix
    
    #create and reshape H matrix to a 3x3 np array
    H = null_space(A)
    H = H.reshape(3,3)
    last = H[2][2]            
    H = H * (1/last)        #normalize H 
    #------------------

    return H, A
