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