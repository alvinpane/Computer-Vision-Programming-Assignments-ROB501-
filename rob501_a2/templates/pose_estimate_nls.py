import numpy as np
from numpy.linalg import inv, norm
#from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

import numpy as np
from numpy.linalg import inv

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    #--- FILL ME IN ---
 
    # Code goes here...
    
    """
    Yaw = np.array([[np.cos(psi), -1*np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi),    0],
                   [0,         0,          1]])
                   
    Pitch = np.array([[np.cos(theta)       , 0       ,        np.sin(theta) ],
                   [0                , 1       ,        0          ],
                   [-1*np.sin(theta)    , 0       ,       np.cos(theta)  ]])
                   
    Roll = np.array([[1       , 0       , 0             ],
                   [0       , np.cos(phi),    -1*np.sin(phi)],
                   [0       , np.sin(phi),       np.cos(phi)]])
    
    Cwc = Yaw @ Pitch @ Roll"""
    
    """Process
    Part 1
- Find individual rotation matrices r,p,y
-determine Jacobian for each of this 3 rotation matrices using lecture 8 slide 28 
Part 2
- find translation jacobian for xyz
Part 3
- once you have all 6 jacobian x y z r p y, combine them into 3*6 matrix 
- now we use the formula to get ride of third row of this jacobian 
- the final value for Jacobean is the output"""
    
    
    ###PART 1
    #Extract rotation matrix,translation vector, world pt coords, and camera intrins parameters
    R = Twc[0:3,0:3]
    
    t = Twc[0:3,3]
    
    tx=t[0]
    ty=t[1]
    tz=t[2]
    
    WR=Wpt.reshape(3)
    wx=WR[0]
    wy=WR[1]
    wz=WR[2]

    flatK = K[0,:].flatten()
    flatK2 = K[1,1:].flatten()
    fx = flatK[0]
    s= flatK[1]
    cx=flatK[2]
    fy = flatK2[0]
    cy=flatK2[1]

    
    #compose inverse of the transform -R^(-1)*t (inverse of the transform), and find euler angles
    flatR = R.flatten()
    r1=flatR[0]
    r2=flatR[1]
    r3=flatR[2]
    r4=flatR[3]
    r5=flatR[4]
    r6=flatR[5]
    r7=flatR[6]
    r8=flatR[7]
    r9=flatR[8]
    yaw = np.arctan2(r4,r1)                           
    pitch = np.arctan2(-1*r7,np.sqrt(1-r7 ** 2))        
    roll = np.arctan2(r8,r9)        
    #inverse transform composition
    p1 = -1 * (r1 * tx + r4 * ty + r7 * tz)
    p2 = -1 * (r2 * tx + r5 * ty + r8 * tz)
    p3 = -1 * (r3 * tx + r6 * ty + r9 * tz)
        
    #on image
    ix_pro = wx*(fx*r1 + s*r2 + cx*r3) + wy*(fx*r4 + s*r5 + cx*r6) + wz*(fx*r7 + s*r8 + cx*r9) + fx*p1 + s*p2 + cx*p3
    iy_pro = wx*(fy*r2 + cy*r3) + wy*(fy*r5 + cy*r6) + wz*(fy*r8 + cy*r9) + fy*p2 + cy*p3

    ###PART 2
    
    ## A) First we find derivative of translation vector tx ty tz with respect to each component
    
    #step 1, compute dx/dtx, dy/dtx
    denom = r3 * wx + r6 * wy + r9 * wz + p3
    deriv_denom = -r3
    deriv_num = (fx * -1 * r1) + (s * -1 * r2) + (cx * -1 * r3)
    deriv_num2 = (fy * -1 * r2) + (cy * -1 * r3)
    dx_dtx = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    dy_dtx = (denom * deriv_num2 - deriv_denom * iy_pro) / (denom ** 2)
    
    #step 2, compute dx/dty, dy/dty
    deriv_denom = -r6
    deriv_num = (fx * -1 * r4) + (s * -1 * r5) + (cx * -1 * r6)
    deriv_num2 = (fy * -1 * r5) + (cy * -1 * r6)
    dx_dty = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    dy_dty = (denom * deriv_num2 - deriv_denom * iy_pro) / (denom ** 2)
    
    #step 3, compute dx/dtz, dy/dtz
    deriv_denom = -r9
    deriv_num = (fx * -1 * r7) + (s * -1 * r8) + (cx * -1 * r9)
    deriv_num2 = (fy * -1 * r8) + (cy * -1 * r9)
    dx_dtz = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    dy_dtz = (denom * deriv_num2 - deriv_denom * iy_pro) / (denom ** 2)
    
    ## B) Now find derivative on Rot matrix wrt to roll, pitch, yaw
    
    phi,theta,psi = roll,pitch,yaw
    
    #Encode matrices for Yaw pitch and Roll

    CYaw = np.array([[np.cos(psi), -1*np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi),    0],
                   [0,         0,          1]])
                   
    CPitch = np.array([[np.cos(theta)       , 0       ,        np.sin(theta) ],
                   [0                , 1       ,        0          ],
                   [-1*np.sin(theta)    , 0       ,       np.cos(theta)  ]])
                   
    CRoll = np.array([[1       , 0       , 0             ],
                   [0       , np.cos(phi),    -1*np.sin(phi)],
                   [0       , np.sin(phi),       np.cos(phi)]])
    
    # step 1, dx/dtyaw, dy/dtyaw
    dYaw = np.array([[-1*np.sin(psi), -1*np.cos(psi),    0],
                    [np.cos(psi)   , -1*np.sin(psi),    0],
                    [0          ,           0,    0]])
    
    
    dRoll = dYaw.dot(CPitch.dot(CRoll))
    flatdR = dRoll.flatten()
    dr1=flatdR[0]
    dr2=flatdR[1]
    dr3=flatdR[2]
    dr4=flatdR[3]
    dr5=flatdR[4]
    dr6=flatdR[5]
    dr7=flatdR[6]
    dr8=flatdR[7]
    dr9=flatdR[8]
    
    dp1 = -1*(tx*dr1 +ty*dr4 +tz*dr7) 
    dp2 = -1*(tx*dr2 +ty*dr5 +tz*dr8) 
    dp3 = -1*(tx*dr3 +ty*dr6 +tz*dr9) 
    deriv_denom = dr3 * wx + dr6 * wy + dr9 * wz + dp3
    deriv_num = wx*(fx  * dr1 + s * dr2 + cx * dr3) + wy*(fx*dr4 + s*dr5 + cx*dr6) + wz*(fx*dr7 + s*dr8 + cx*dr9) + fx * dp1 + s * dp2 + cx * dp3
    deriv_num2 = wx*(fy * dr2 + cy * dr3) + wy*(fy*dr5 + cy*dr6) + wz*(fy*dr8 + cy*dr9) + fy * dp2 + cy * dp3
    dx_dtq = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    dy_dtq = (denom * deriv_num2 - deriv_denom * iy_pro) / (denom ** 2)
    
    #step 2, dx/dtroll, dy.dtroll
    #encode matrix, manually derived
    dRoll = np.array([[0       , 0        , 0             ],
                   [0       ,-1*np.sin(phi),    -1*np.cos(phi)],
                   [0       ,   np.cos(phi),    -1*np.sin(phi)]])
    
    dr = CYaw.dot(CPitch.dot(dRoll))
    flat_dR = dr.flatten()
    dr1=flat_dR[0]
    dr2=flat_dR[1]
    dr3=flat_dR[2]
    dr4=flat_dR[3]
    dr5=flat_dR[4]
    dr6=flat_dR[5]
    dr7=flat_dR[6]
    dr8=flat_dR[7]
    dr9=flat_dR[8]
    
    dp1 = -1*(tx*dr1 +ty*dr4 +tz*dr7) 
    dp2 = -1*(tx*dr2 +ty*dr5 +tz*dr8) 
    dp3 = -1*(tx*dr3 +ty*dr6 +tz*dr9) 
    
    deriv_denom = dr3 * wx + dr6 * wy + dr9 * wz + dp3
    deriv_num = wx*(fx  * dr1 + s * dr2 + cx * dr3) + wy*(fx*dr4 + s*dr5 + cx*dr6) + wz*(fx*dr7 + s*dr8 + cx*dr9) + fx * dp1 + s * dp2 + cx * dp3
    deriv_num2 = wx*(fy * dr2 + cy * dr3) + wy*(fy*dr5 + cy*dr6) + wz*(fy*dr8 + cy*dr9) + fy * dp2 + cy * dp3
    dx_dtr = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    dy_dtr = (denom * deriv_num2 - deriv_denom * iy_pro) / (denom ** 2)
    
    # step 3, dx/dtpitch, dy/dtroll
    dPitch = np.array([[-1*np.sin(theta)       , 0       ,        np.cos(theta) ],
                   [0                    , 0       ,        0          ],
                   [-1*np.cos(theta)        , 0       ,    -1*np.sin(theta)  ]])
    dr = CYaw.dot(dPitch.dot(CRoll))
    #print(dYaw)
    #print(dcp)
    #print(CRoll)
    flat_dR = dr.flatten()
    dr1=flat_dR[0]
    dr2=flat_dR[1]
    dr3=flat_dR[2]
    dr4=flat_dR[3]
    dr5=flat_dR[4]
    dr6=flat_dR[5]
    dr7=flat_dR[6]
    dr8=flat_dR[7]
    dr9=flat_dR[8]
    
    dp1 = -1*(tx*dr1 +ty*dr4 +tz*dr7) 
    dp2 = -1*(tx*dr2 +ty*dr5 +tz*dr8) 
    dp3 = -1*(tx*dr3 +ty*dr6 +tz*dr9) 
    deriv_denom = dr3 * wx + dr6 * wy + dr9 * wz + dp3
    #print(denom)
    #print(deriv_denom)
    deriv_num = wx*(fx  * dr1 + s * dr2 + cx * dr3) + wy*(fx*dr4 + s*dr5 + cx*dr6) + wz*(fx*dr7 + s*dr8 + cx*dr9) + fx * dp1 + s * dp2 + cx * dp3
    dx_dtp = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    deriv_num2 = wx*(fy * dr2 + cy * dr3) + wy*(fy*dr5 + cy*dr6) + wz*(fy*dr8 + cy*dr9) + fy * dp2 + cy * dp3
    dx_dtp = (denom * deriv_num - deriv_denom * ix_pro) / (denom ** 2)
    dy_dtp = (denom * deriv_num2 - deriv_denom * iy_pro) / (denom ** 2)
    
    ### PART 3, compose jacobian
    
    J = np.array([ [dx_dtx, dx_dty, dx_dtz, dx_dtr, dx_dtp, dx_dtq],
                   [dy_dtx, dy_dty, dy_dtz, dy_dtr, dy_dtp, dy_dtq]])

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame."""
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---

    # Some hints on structure are included below...

    # 1. Convert initial guess to parameter vector (6 x 1).
    # ...

    iter = 1
   
    
    #error threshold
    eps = 0.0000001
    #get the original parameters
    params = epose_from_hpose(Twc_guess) 
    Twc = hpose_from_epose(params)
    
    # 2. Main loop - continue until convergence or maxIters.
    while iter <= maxIters:
        # 3. Save previous best pose estimate.
        # ...
        params_prev = params
        curr_Twc = hpose_from_epose(params_prev)
        curr_R = curr_Twc[0:3,0:3]
        curr_t = curr_Twc[0:3,3:4]
        
         # 4. Project each landmark into image, given current pose estimate.
        for i in np.arange(tp):
       
            #get jacobian for current point
            WPT=Wpts[0:3,i:i+1]
            J_curr = find_jacobian(K, curr_Twc, WPT)
            
            #update J
            J[2*i:2*i+2,0:6] = J_curr
            
            #current landmark estimate
            CLE = K.dot(curr_R.T).dot(WPT-curr_t)
            
            #error for current landmark estimate, and update residual
            err=(CLE/CLE[2,0])[0:2,0:1] - Ipts[0:2,i:i+1]
            dY[2*i:2*i+2,0:1]=err
        
        
            pass
       
        #for i in np.arange(tp):
           # pass

        # 5. Solve system of normal equations for this iteration.
        # ...
        JT = J.T
        #dx = np.dot(np.dot( inv(np.dot(J.T,J)),J.T),dY)
        dx = -1*inv(JT.dot(J)).dot(JT).dot(dY)
        #update params in direction of dx
        
        params = params_prev + dx
        
        Twc = hpose_from_epose(params)

        # 6. Check - converged?
        diff = norm(params - params_prev)

        if norm(diff) < 1e-12:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
        iter += 1

    # 7. Compute and return homogeneous pose matrix Twc.
    
    #------------------

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc