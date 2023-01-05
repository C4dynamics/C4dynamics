import numpy as np

import C4dynamics as c4d

class rigidbody(c4d.datapoint):  # 
  """ 
    the rigidbody object is the most basic element in the rotational dynamics domain.
    a rigidbody object is also a datapoint. 
  """
  
  # 
  # 
  ##
  phi   = 0
  theta = 0
  psi   = 0
  p     = 0
  q     = 0
  r     = 0
  p_dot = 0
  q_dot = 0
  r_dot = 0
  phi0   = 0
  theta0 = 0
  psi0   = 0
  dcm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  


  def __init__(obj, **kwargs):
    obj.__dict__.update(kwargs)

    obj.phi0   = obj.phi
    obj.theta0 = obj.theta
    obj.psi0   = obj.psi
   
   
  def inertial_from_body_dcm(obj): # bound method 
    # Bound methods have been "bound" to an instance, and that instance will be passed as the first argument whenever the method is called.
    return pri.dcm321(obj.phi, obj.theta, obj.psi)

 
 
  @staticmethod
  def eqm(xin, L, M, N, ixx, iyy, izz):
    import math as m
    phi   = xin[0]
    theta = xin[1]
    psi   = xin[2]
    p     = xin[3]
    q     = xin[4]
    r     = xin[5]
    pdot  = xin[6]
    qdot  = xin[7]
    rdot  = xin[8]
    
    dphi   = (q * m.sin(phi) + r * m.cos(phi)) * m.tan(theta)
    dtheta =  q * m.cos(phi) - r * m.sin(phi)
    dpsi   = (q * m.sin(phi) + r * m.cos(phi)) / m.cos(theta)
    dp     = (L - q * r * (izz - iyy)) / ixx
    dq     = (M - p * r * (ixx - izz)) / iyy
    dr     = (N - p * q * (iyy - ixx)) / izz
    
    return phi, dtheta, dpsi, dp, dq, dr
   
      

 