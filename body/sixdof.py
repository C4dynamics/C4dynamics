import numpy as np

import C4dynamics as c4d

class sixdof():
  """ 
    sixdof class provides an object for running 6 degrees of freedom simulation for 
        an object of a C4dynamics.rigidbody type. 
  """
  
  # 
  # this class holds only the current state and a function to update the state according to the equations of motion 
  ##
  
  _y = np.zeros((1, 10))

  def __init__(obj, rb):
    obj.y = [rb.x, rb.y, rb.z, rb.vx, rb.vy, rb.vz, rb.phi, rb.theta, rb.psi, rb.p, rb.q, rb.r]
   
  def run(obj, t0, tf): 
    phi   = xin[0]
    theta = xin[1]
    psi   = xin[2]
    p     = xin[3]
    q     = xin[4]
    r     = xin[5]
    

  @staticmethod
  def eqm(xin, fx, fy, fz, m):    
    # 
    # see military handbook for missile flight simulation ch.12 simulation synthesis
    ##
    
    # 
    # dynamic pressure
    ## 
    Q = 1/2 * rho * v**2
    
    
    # 
    # aerodynamic forces
    ##
    
    # lift and drag
    cL = cLa * alpha_total
    L = Q * s * cL 
    
    cD = cD0 + k * cL**2
    D = Q * s * cD
    
    # in body frame
    A = D * np.cos(alpha_total) - L * np.sin(alpha_total)
    N = D * np.sin(alpha_total) + L * np.cos(alpha_total)
    
    fAxb = -A
    fAyb =  N * (-v / np.sqrt(v**2 + w**2))
    fAzb =  N * (-w / np.sqrt(v**2 + w**2))

    # 
    # aerodynamic moments 
    ## 

    cMref = cMa * alpha + cMd * dpitch
    cNref = cNb * beta  + cNd * dyaw
    
    
    # wrt center of mass
    cM = cMref - cNz * (xcm - xref) / d + d / (2 * v) * (cMq + cMadot) * q
    cN = cNref - cNy * (xcm - xref) / d + d / (2 * v) * (cNr + cNbdot) * r
    
    mA = Q * cM * s * d
    nA = Q * cN * s * d
    
    
    # 
    # gravity
    ## 
    fGe = [[0], [0], [m * g]]
    fGb = BI * fGe 

    
    
    #
    # translational motion derivatives
    ##
    du = (fAxb + fGxb) / m - (q * w - r * v)
    dv = (fAyb + fGyb) / m - r * u
    dw = (fAzb + fGzb) / m + q * u
    
    
    
    # 
    # euler angles derivatives 
    ## 
    
    dphi   = (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    dtheta =  q * np.cos(phi) - r * np.sin(phi)
    dpsi   = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    # 
    # angular motion derivatives 
    ## 
    dp     = (L - q * r * (izz - iyy)) / ixx
    dq     = (M - p * r * (ixx - izz)) / iyy
    dr     = (N - p * q * (iyy - ixx)) / izz
    
     
    return dphi, dtheta, dpsi, dp, dq, dr
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   




