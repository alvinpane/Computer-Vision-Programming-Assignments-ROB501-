import numpy as np
from numpy.linalg import inv, lstsq

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
    coeff=Soln[0]
    alpha,beta,gamma,delta,epsilon,zeta  = coeff.T[0]
    
    #print(alpha,beta,gamma,delta,epsilon, zeta)
    #once we have the coefficients, we can compose the solution
    
    v1 = [2*alpha, beta]
    v2 = [beta, 2*gamma]
    b1= [delta, epsilon]
    
    M=np.array([v1,v2])
    print(M)
    x=np.array([[b1[0]],[b1[1]]])
    M_inv= inv(M)
    new_M = -1*M_inv
    print(x)
    print(new_M)
    #print(x)
    pt = (new_M).dot(x)
    
    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt