import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline. All arrays should contain float64 values.

    Parameters:
    -----------
    Kl   - 3x3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3x3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    # Compute baseline (right camera translation minus left camera translation).

    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
 
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.

    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Add code here...
    #************************************************** 
    #rotation matrices
    Rl = Twl[0:3,0:3]
    Rr = Twr[0:3,0:3]
    #position vectors
    p_l = Twl[0:3,3]
    p_r = Twr[0:3,3]
    #compute unit vectors from optical center to image plane points, rayl and rayr
    
    #temp = inv(Kl) @ np.vstack([pl,[1]]) 
    rl = Rl @ inv(Kl) @ np.vstack([pl,1])
    rayl = rl / norm(rl)
    
    #temp = inv(Kr) @ np.vstack([pr,1]) 
    rr = Rr @ inv(Kr) @ np.vstack([pr,1])
    rayr = rr/norm(rr)
    
    #compute baseline, subtract translations
    baseline = p_r - p_l
    
    #compute projected segment lengths
    temp = baseline @ rayl - (baseline @ rayr * rayl.T @ rayr)
    ml = temp / (1 -  (rayl.T @ rayr)**2 )
    mr = rayl.T @ rayr * ml - baseline @ rayr
    
    
    #segment endpoints Pl, Pr
    Pl = Twl[0:3,3:4] +rayl*ml
    Pr = Twr[0:3,3:4] + rayr*mr
    
    ##fill in with ray Jacobians
    
    
    ##compose dRl/du and dRl/du, partials of Rl with respect to u and v
    temp1 = Twl[0:3,0:3] @ inv(Kl)[0:3,0:1]    #these are the numerators
    temp2 = Twl[0:3,0:3] @ inv(Kl)[0:3,1:2]
    #denominators
    temp3 = (1/2)*((norm(rl) ** 2) ** (-1/2)) *(2*rl[0,0]*temp1[0,0] +  2*rl[1,0]*temp1[1,0] + 2*rl[2,0]*temp1[2,0])
    temp4 = (1/2)*((norm(rl) ** 2) ** (-1/2)) *(2*rl[0,0]*temp2[0,0] + 2*rl[1,0]*temp2[1,0] + 2*rl[2,0]*temp2[2,0])
    
    dRl_dul = (temp1 * norm(rl) - rl * temp3) / (norm(rl) ** 2)
    dRl_dvl = (temp2 * norm(rl) - rl * temp4) / (norm(rl) ** 2)
    
    #the partials of Rl with respect ur and vr are the same, simply 0 vectors
    dRl_dur = np.zeros((3,1))
    dRl_dvr = dRl_dur
    
    #now, with the above partials, we can put together drayl 
    drayl = np.hstack([dRl_dul,dRl_dvl,dRl_dur,dRl_dvr])
    
    ##repeat the above process for Rr
    
    t1 = Twr[0:3,0:3] @ inv(Kr)[0:3,0:1] 
    t2 = Twr[0:3,0:3] @ inv(Kr)[0:3,1:2]
    t3 = (1/2)*((norm(rr) ** 2 ) ** (-1/2)) *(2 * rr[0,0] * t1[0,0] +  2*rr[1,0]*t1[1,0] + 2*rr[2,0]*t1[2,0])
    t4 = (1/2)*((norm(rr) ** 2 ) ** (-1/2)) *  (2 * rr[0,0] * t2[0,0] +  2 * rr[1,0]*t2[1,0] + 2*rr[2,0]*t2[2,0])
    
    #dul and dvl
    dRr_dur = (t1 * norm(rr) - rr * t3) / (norm(rr) ** 2)
    dRr_dvr = (t2 * norm(rr) - rr * t4) / (norm(rr) ** 2)
        
    #the partials of Rr with respect ul and vl are the same, simply 0 vectors
    dRr_dul = np.zeros(shape = (3,1))
    dRr_dvl = dRr_dul
    
    #compose drayr with the above partials
    drayr = np.hstack([dRr_dul,dRr_dvl,dRr_dur,dRr_dvr])
  
    
    b = baseline
    #**************************************************
    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    #--- FILL ME IN ---

    # 3D point.

    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    #**************************************************
    P = (Pl + Pr)/2
    
    cov = np.zeros((4,4))
    cov[0:2, 0:2] = Sl
    cov[2:4, 2:4] = Sr
    #cov = np.hstack([Sl, np.zeros((2,2))])
    #cov = np.vstack( [cov, np.hstack([np.zeros((2,2)), Sr ])] )
    
    S= JP @ cov @ JP.T
    
    #**************************************************
    #------------------

    # Check for correct outputs...
    correct = isinstance(Pl, np.ndarray) and Pl.shape == (3, 1) and \
              isinstance(Pr, np.ndarray) and Pr.shape == (3, 1) and \
              isinstance(P,  np.ndarray) and P.shape  == (3, 1) and \
              isinstance(S,  np.ndarray) and S.shape  == (3, 3)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Pl, Pr, P, S