import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.

#DLT Homography, code from ROB501 assignment 1
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

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    # Code goes here.
    rows = I.shape[0]  
    cols = I.shape[1]    #get dimensions for A and b, A is m*n by 6 and b is m*n by 1
                            # (x^2,xy,y^2,x,y,1) -> 6 elements
    
    dim = rows*cols
    #A_dim = (dim,6)
    
    
    A=np.zeros((dim,6)) #initialize A matrix to empty
    b=np.zeros((dim,1)) #initialize b vector to empty
    
    #now we need to populate the b vector with the image points
    i = 0
    for y in range(rows):
        for x in range(cols):
            b[i,0] = I[y,x]
            i=i+1
            
    #do the same for A matrix
    j=0
    for y in range(rows):
        for x in range(cols):
            A[j,:] = [x*x, x*y, y*y, x, y, 1]
            j=j+1
            
    #print(A)
    #print(b)
    
    Soln = lstsq(A,b,None)
    coeffs=Soln[0]
    alpha,beta,gamma,delta,epsilon,zeta  = coeffs.T[0]
    
    #print(alpha,beta,gamma,delta,epsilon, zeta)
    #once we have the coefficients, we can compose the solution
    
    v1 = [2*alpha, beta]
    v2 = [beta, 2*gamma]
    b1= [delta, epsilon]
    
    M=np.array([v1,v2])
    #print(M)
    x=np.array([[b1[0]],[b1[1]]])
    M_inv= inv(M)
    new_M = -1*M_inv
    #print(x)
    #print(new_M)
    #print(x)
    pt = (new_M).dot(x)
    
    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---
    """For part 2, we are given a bpoly that contains 4 points which are the corners of the blue rectangular box if you run the learner example for visualization; as well as Wpts that includes the coordinates of the 48 points in 3D. Note that since those 48 points are coplanar, their z values don't matter, which effectively makes them 2D. 
 
Here we can try to apply what you mentioned, which is to find the homography between two set of points. Obiviously we need to locate the correponding points of the 4 bpoly corners in 3D, and we can do so by estimation based on the 4 extreme points of Wpts. Note that you are also given the side length of a square is 63.5mm in the assignment handout. You can use that with some scaling to estimate the 3D location of the corners."""

    # Code goes here...
    
    rows = I.shape[0]  #m
    cols = I.shape[1]  #n
    
    #Want to locate points of 4 bpoly corners, we are given the side length of a square is 63.5mm, we can scale this by 1000 to get the distance in m(note we dont care about z)
    d = 63.5 / 1000.0
    #scale for x border
    x_b = (1.0/3.0) * d 
    #scale for y border
    y_b = (1.0/5.0) * d 
    #find delta from junctions to border 
    dx = x_b + d
    dy = y_b + d
    
    #note that the above is in world points
    
    #find points of the corners, C1 is top left, C2 is top right, C3 is bottom left, C4 is bottom right
    
    C1 = Wpts[:,0:1]+ np.array([[-1*dx], [-1*dy],[0]])
    C2 = Wpts[:,7:8] + np.array([[dx], [-1*dy],[0]])
    C3 = Wpts[:,-8:-7] + np.array([[-1*dx], [dy],[0]])
    C4 = Wpts[:,-1:]+ np.array([[dx], [dy],[0]])
    combined = np.hstack([C1,C2,C4,C3])[:2,:]
    #print(combined)
    #use the homography process from assignment 1 to find transformation between the board frame and the distorted board in the picture
    H,A = dlt_homography(combined,bpoly)
    Wpts[2,:] = 1
    junctions = (H.dot(Wpts) / H.dot(Wpts)[2,:])[:2,:]  #
    #print(junctions)
    #Then use the saddle point function to find precise location in a patch around the predicted location
    results_list = []
    w = 8 #use some window around predicted points
    JT=junctions.T 
    for index, row in enumerate(JT):
        x,y =row
        x,y = int(x), int(y)
        #print(x)
        #print(y)
        #print(I[y-w:y+w,x-w:x+w])
        point_of_interest = saddle_point(I[y-w:y+w,x-w:x+w])
        results_list.append( [x - w + point_of_interest[0,0], y - w + point_of_interest[1,0]] )
    
   
    Ipts = np.array(results_list).T

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts