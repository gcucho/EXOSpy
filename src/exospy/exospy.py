import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.special import gamma
import math
import scipy as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm
from scipy.integrate import quad
from scipy import interpolate


class exosgrid:
  def __init__(self):
    rvals = []
    pvals = []
    tvals = []
    rmin  = [] 
    rmax  = []
    rstep = []
    pmin  = []
    pmax  = []
    pstep = []
    tmin  = []
    tmax  = []
    tstep = []
    numR  = []
    numP  = []
    numT  = []
  # Calculate internal values for exospheric grid (spherical)
  def calculate_intvals(self):
    self.numR  = (self.rmax-self.rmin)/self.rstep;
    self.numP  = (self.pmax-self.pmin)/self.pstep;
    self.numT  = (self.tmax-self.tmin)/self.tstep;
    self.numvoxels  = self.numR*self.numP*self.numT;
    self.rvals      = np.arange(self.rmin,self.rmax+self.rstep,self.rstep,dtype=np.float64)
    self.pvals      = np.arange(self.pmin,self.pmax+self.pstep,self.pstep,dtype=np.float64)
    self.tvals      = np.arange(self.tmin,self.tmax+self.tstep,self.tstep,dtype=np.float64)

#---------------------- H DENSITY MODELS ---------------------------------------

def AB_coefficients(model,radius):
  #num_coefficients = radius.shape[0]
  coefficients = np.zeros([20,1])#num_coefficients])

  if model == 'Z15MIN':
    a = np.array([1, 938.00, 92.20, -385.26, 2042.26, -421.34, 0, 0, 0, 0]) * 10**(-4);
    a[0] = 1
    b = np.array([0, 135.41, 198.16, -597.06, -916.65, 1196.10, 0, 0, 0, 0]) * 10**(-4);
    c = np.array([0, 0, 4870.41, 0, -2506.10, 2783.95, 0, 0, 0, 0]) * 10**(-4);
    d = np.array([0, 0, -2632.88, 0, 1578.28, -1331.32, 0, 0, 0, 0]) * 10**(-4);
    f = np.log(radius);

  if model == 'Z15MAX':
    a = np.array([1, -921.29, 6763.12, -494.96, -284.02, -556.96, 0, 0, 0, 0]) * 10**(-4);
    a[0] = 1
    b = np.array([0, 790.11, -3088.94, -405.36, 44.03, 1303.13, 0, 0, 0, 0]) * 10**(-4);
    c = np.array([0, 0, 1289.94, 0, -753.84, 2029.43, 0, 0, 0, 0]) * 10**(-4);
    d = np.array([0, 0, -788.54, 0, 256.56, -1084.30, 0, 0, 0, 0]) * 10**(-4);
    f = np.log(radius);

  #for i in range(num_coefficients):
  coefficients[0:10,0] = a + b*f#[i];
  coefficients[10:20,0] = c + d*f#[i];

  return coefficients

#-------------------------------------------------------------------------------
def N_coefficients(model,radius):
  coefficient = 0#np.zeros([radius.shape[0],1])

  if model == 'Z15MIN':
    coefficient = 12264.1*radius**(-2.87646);

  if model == 'Z15MAX':
    coefficient = 16840.9*radius**(-2.74640);

  return coefficient

#-------------------------------------------------------------------------------
def spherical_harmonics(theta):

  Y_lm = np.zeros([10,1])

  Y_lm[0,:] = np.sqrt(1/(4*np.pi)) #Y_00
  Y_lm[1,:] = np.sqrt(3/(4*np.pi))*np.cos(theta) #Y_10
  Y_lm[2,:] = -np.sqrt(3/(8*np.pi))*np.sin(theta) #Y_11
  Y_lm[3,:] = np.sqrt(5/(4*np.pi))*(3/2*np.cos(theta)**2-1/2) #Y_20
  Y_lm[4,:] = -np.sqrt(15/(8*np.pi))*np.sin(theta)*np.cos(theta) #Y_21
  Y_lm[5,:] = 1/4*np.sqrt(15/(2*np.pi))*np.sin(theta)**2 #Y_22
  Y_lm[6,:] = np.sqrt(7/(4*np.pi))*(5/2*np.cos(theta)**3-3/2*np.cos(theta)) #Y_30
  Y_lm[7,:] = -1/4*np.sqrt(21/(4*np.pi))*np.sin(theta)*(5*np.cos(theta)**2-1) #Y_31
  Y_lm[8,:] = 1/4*np.sqrt(105/(2*np.pi))*np.sin(theta)**2.*np.cos(theta) #Y_32
  Y_lm[9,:] = -1/4*np.sqrt(35/(4*np.pi))*np.sin(theta)**3 #Y_33

  return Y_lm

#-------------------------------------------------------------------------------
def get_density(model, radius, theta, phi):
    """
    Backward-compatible wrapper.

    Inputs come as 1x1 arrays:
      radius : Re
      theta  : colatitude in radians (0..pi)   <-- your exosgrid.tvals are 0..180 deg
      phi    : longitude in radians (0..2pi)

    Returns scalar float (cm^-3).
    """
    m = str(model).lower().strip()

    # unwrap 1x1 arrays
    r_re  = float(np.asarray(radius).ravel()[0])
    colat = float(np.asarray(theta).ravel()[0])   # already colatitude (rad)
    lon   = float(np.asarray(phi).ravel()[0])     # longitude (rad)

    # normalize model name (optional but robust)
    model_map = {
        "bailey": "bailey_2008",
        "bailey2008": "bailey_2008",
        "bailey_2008": "bailey_2008",

        "zoennchen2015_min": "zoennchen_2015_min",
        "zoennchen_2015_min": "zoennchen_2015_min",
        "z2015_min": "zoennchen_2015_min",

        "zoennchen2015_max": "zoennchen_2015_max",
        "zoennchen_2015_max": "zoennchen_2015_max",
        "z2015_max": "zoennchen_2015_max",

        "zoennchen2024_2008": "zoennchen_2024_2008",
        "zoennchen_2024_2008": "zoennchen_2024_2008",
        "2008": "zoennchen_2024_2008",

        "zoennchen2024_2013": "zoennchen_2024_2013",
        "zoennchen_2024_2013": "zoennchen_2024_2013",
        "2013": "zoennchen_2024_2013",

        "zoennchen2024_2015": "zoennchen_2024_2015",
        "zoennchen_2024_2015": "zoennchen_2024_2015",
        "2015": "zoennchen_2024_2015",
    }
    model_name = model_map.get(m, m)

    # Call your new 6-model dispatcher (recommended)
    n = h_density(
        model_name,
        r=r_re,
        theta=colat,
        phi=lon,
        degrees=False,
        r_units="Re"
    )
    return float(np.asarray(n))
#def get_density(model,radius,theta,phi):
#  # Verifying they are column vectors
#  #if theta.shape[1] > theta.shape[0]:
#  #  theta = np.transpose(theta)

#  #if phi.shape[1] > phi.shape[0]:
#  #  phi = np.transpose(phi)

#  #if radius.shape[1] > radius.shape[0]:
#  #  radius = np.transpose(radius)

#  n_h = 0 
#  n_radii = 1#radius.shape[0]

#  N     = np.zeros([n_radii,1])
#  A_lm  = np.zeros([10,n_radii])
#  B_lm  = A_lm
#  Y_lm  = 0 #np.zeros([theta.shape[0],1])

#  N     = N_coefficients(model,radius)
#  AB    = AB_coefficients(model,radius); 
#  Y_lm  = spherical_harmonics(theta);

#  A_lm  = AB[0:10,:]
#  B_lm  = AB[10:20,:] 

#  #l = 0, m = 0
#  n_h = n_h + (A_lm[0,:]*np.cos(0*phi)+B_lm[0,:]*np.sin(0*phi))*Y_lm[0,:]
#  #l = 1, m = 0, 1
#  n_h = n_h + (A_lm[1,:]*np.cos(0*phi)+B_lm[1,:]*np.sin(0*phi))*Y_lm[1,:] + (A_lm[2,:]*np.cos(1*phi)+B_lm[2,:]*np.sin(1*phi))*Y_lm[2,:]
#  #l = 2, m = 0, 1
#  n_h = n_h + (A_lm[3,:]*np.cos(0*phi)+B_lm[3,:]*np.sin(0*phi))*Y_lm[3,:] + (A_lm[4,:]*np.cos(1*phi)+B_lm[4,:]*np.sin(1*phi))*Y_lm[4,:]
#  #l = 2, m = 2
#  n_h = n_h + (A_lm[5,:]*np.cos(2*phi)+B_lm[5,:]*np.sin(2*phi))*Y_lm[5,:]
#  #l = 3, m = 0, 1
#  n_h = n_h + (A_lm[6,:]*np.cos(0*phi)+B_lm[6,:]*np.sin(0*phi))*Y_lm[6,:] + (A_lm[7,:]*np.cos(1*phi)+B_lm[7,:]*np.sin(1*phi))*Y_lm[7,:]
#  #l = 3, m = 2, 3
#  n_h = n_h + (A_lm[8,:]*np.cos(2*phi)+B_lm[8,:]*np.sin(2*phi))*Y_lm[8,:] + (A_lm[9,:]*np.cos(3*phi)+B_lm[9,:]*np.sin(3*phi))*Y_lm[9,:]

#  density = n_h*N*np.sqrt(4*np.pi);

#  return density

#----- DOLON's contribution ----------------------------------------------------
def func(r,a,b,c,d):
    return (a*np.exp(b*r)) + (c*np.exp(d*r))

#----- DOLON's contribution ----------------------------------------------------
def partition_escape(lamda_r,lamda_rc):
    psi_1  = lamda_r**2/(lamda_r+lamda_rc)
    gamma1 = gammainc(1.5,lamda_r)*gamma(1.5) 
    gamma2 = gammainc(1.5,lamda_r-psi_1)*gamma(1.5)
    
    zeta_esc = (1./math.sqrt(math.pi))* (gamma(1.5) - gamma1 - \
                ((np.sqrt(lamda_rc**2 - lamda_r**2)/lamda_rc)*\
                np.exp(-psi_1)* (gamma(1.5)-gamma2)))
    zeta_esc[0] = 0.
    
    return zeta_esc

#----- DOLON's contribution ----------------------------------------------------
def partition_ballistic(lamda_r,lamda_rc):
    psi_1  = lamda_r**2/(lamda_r+lamda_rc)
    gamma1 = gammainc(1.5,lamda_r)*gamma(1.5)
    gamma2 = gammainc(1.5,lamda_r-psi_1)*gamma(1.5)
    zeta_bal = (2./math.sqrt(math.pi))*(gamma1-((np.sqrt(lamda_rc**2-lamda_r**2)/lamda_rc)\
                             *np.exp(-psi_1)*gamma2))
    zeta_bal[0] = 0
    
    return zeta_bal

#----- DOLON's contribution ----------------------------------------------------
def chamberlain(exo_dens,exo_temp,alt_val):
    
    target_radius = 6371.0084          #Earth radius in km
    planet_mass   = 5.972E27           #Mass of Earth in gm
    Grav          = 6.6738400E-8        #Universal gravitational constant in CGS 
    kb            = 1.3806488E-23       #Boltzmann constant in MKS
    mp            = 1.6722178E-24       #Mass of an H atom in gm
    
    exo_ht        = 480*1.E5 + target_radius*1.E5
    r_c           = exo_ht
    alt_val       = alt_val*1.E5 + target_radius*1.E5

    lamda_rc       = (Grav*planet_mass*mp)/(kb*1.E7*exo_temp*r_c)
    lamda_r        = (Grav*planet_mass*mp)/(kb*1.E7*exo_temp*alt_val)
    ballistic_part = partition_ballistic(lamda_r,lamda_rc)
    escape_part    = partition_escape(lamda_r,lamda_rc)

    tot_part_fn      = ballistic_part + escape_part
    num              = np.size(alt_val)
    h_density        = np.zeros(shape=num,dtype=float)
    if (abs(alt_val[0]-exo_ht) < 1.e-3):
        h_density[0]     = exo_dens
        h_density[1:num] = exo_dens*np.exp(-(lamda_rc-lamda_r[1:num]))*tot_part_fn[1:num]
    else:
        h_density = exo_dens*np.exp(-(lamda_rc-lamda_r))*tot_part_fn
    
    return h_density

#--------------- CODE FOR VOXEL/LOS INTERSECTION -------------------------------
def cart2pol(x,y):
  theta = np.arctan2(y,x)
  rho = np.sqrt(x**2+y**2)
  return theta,rho

#-------------------------------------------------------------------------------
def cart2sph(x,y,z):
  xy2 = x**2 + y**2
  radius = np.sqrt(xy2+z**2)
  elev   = np.arctan2(z,np.sqrt(xy2))
  azim   = np.arctan2(y,x)
  return azim,elev,radius

#-------------------------------------------------------------------------------
def sph2cart(azimuth,elevation,r):
  x = r * np.cos(elevation) * np.cos(azimuth)
  y = r * np.cos(elevation) * np.sin(azimuth)
  z = r * np.sin(elevation)
  return x, y, z

#-------------------------------------------------------------------------------
def line_plane_intersection(sat_pos,sat_los,plane):
  # This function allows to calculate the  intersection between a plane and a
  # line. In both, I am using the parametric form. In the case of the line 2
  # points are needed, for the plane, 3 points are needed.
  # The main reference is located in: 
  # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
  # Inputs:
  #       line_or,line_uv: 2 points in the line (origin, unit vector)
  #       plane: [p0;p1;p2], three points to define the plane
  # Output:
  #       out: This is a vector [x,y,z] with the intersection
  #       between the line and plane.

  # function [intersection, bin] = line_plane_intersection(SAT_POS,SAT_LOS,PLANE)
  P0 = plane[0,:]
  P1 = plane[1,:]
  P2 = plane[2,:]

  sat_los = sat_los/np.linalg.norm(sat_los) # unit vector

  [phi1,theta1,rad1] = cart2sph(P0[0],P0[1],P0[2])
  colat1 = 90-theta1*180/np.pi

  [phi2,theta2,rad2] = cart2sph(P1[0],P1[1],P1[2])
  colat2 = 90-theta2*180/np.pi
    
  SAMEToP = 0
  if theta1 == theta2:
    SAMEToP = 1 # equal theta
  
  if phi1 == phi2:
    SAMEToP = 2 # equal phi

  la = sat_pos
  lb = sat_pos + sat_los

  A = np.array([[la[0]-lb[0],P1[0]-P0[0],P2[0]-P0[0]],\
                [la[1]-lb[1],P1[1]-P0[1],P2[1]-P0[1]],\
                [la[2]-lb[2],P1[2]-P0[2],P2[2]-P0[2]]])

  B = np.array([[la[0]-P0[0]],[la[1]-P0[1]],[la[2]-P0[2]]])


  if np.linalg.matrix_rank(A)  == np.min(A.shape):
      invA = np.linalg.inv(A)
      C = invA.dot(B)
      temp = C[0]
      intersection = la + (lb-la) * temp
      bin = 1
      
      if temp<0:
        bin = -1
      
      if np.linalg.norm(intersection-sat_pos) < 1e-5: # same point
        bin = 2

  else:
    intersection = np.array([0,0,0])
    bin = -1

  return intersection, bin

#-------------------------------------------------------------------------------
def line_sph_intersection(sat_pos,sat_los,radius):
  o = sat_pos
  u = sat_los
  c = np.array([0,0,0]) # Earth Center
  r = radius

  temp1 = -(2*u.dot(o-c))
  temp2 = (2*u.dot(o-c))**2 - 4*(np.linalg.norm(u)**2)*(np.linalg.norm(o-c)**2-r**2)
  temp3 = 2*np.linalg.norm(u)**2

  if temp2 < 0: #no solution
    intersection = np.array([0,0,0])
    bin = -2;
    return intersection, bin
  else:
    d1 = (temp1 + np.sqrt(temp2))/temp3
    d2 = (temp1 - np.sqrt(temp2))/temp3;

  if (np.abs(d1) < 1e-10):
    d1 = 0

  if (np.abs(d2)) < 1e-10:
    d2 = 0

  if (d1>=0) and (d2>=0):
    intersection = o + min(d1,d2)*u
    bin = 1
    if np.linalg.norm(intersection-sat_pos) < 1e-5:
      intersection = o + max(d1,d2)*u
      if np.linalg.norm(intersection-sat_pos) < 1e-5:
        bin = 2
        return intersection, bin
      bin = 1
      return intersection, bin
    return intersection, bin

  if (d1<0) and (d2<0): # both solutions are behign the SAT_POS, no intersection
    intersection = np.array([0,0,0])
    bin = -1
    return intersection, bin


  intersection = o + max(d1,d2)*u;
  bin = 1

  if np.linalg.norm(intersection-sat_pos) < 1e-8:
      bin = 2
      return intersection, bin
  return intersection, bin

#-------------------------------------------------------------------------------
def line_cone_intersection(sat_pos,sat_los,th):
  if (th == 90):
    plane = np.array([[0,0,0],[1,0,0],[0,1,0]])
    [intersection,bin] = line_plane_intersection(sat_pos,sat_los,plane)
    return intersection, bin

  a = np.tan(th*np.pi/180)
  b = np.tan(th*np.pi/180)
  A = np.array([a**2,b**2,-(a**2)*(b**2)])

  alpha = A.dot(sat_los**2)
  beta  = 2*A.dot(sat_pos*sat_los)
  gamma = A.dot(sat_pos**2)

  if beta**2  - 4*alpha*gamma < 0: # No solution / no intersection
    intersection = np.array([0,0,0])
    bin = -1
    return intersection, bin
  else:
    t1 = (-beta + np.sqrt(beta**2-4*alpha*gamma))/(2*alpha)
    t2 = (-beta - np.sqrt(beta**2-4*alpha*gamma))/(2*alpha)

  if np.isnan(t1) or np.isnan(t2):
    intersection = np.array([0,0,0])
    bin = -1
    return intersection, bin

    #%%% Analysis for more cases %%%%%%

  if (th < 90):
    intersection1 = sat_pos + t1*sat_los
    intersection2 = sat_pos + t2*sat_los
    # both are over the ecliptic plane
    if intersection1[2] > 0 and intersection2[2] > 0:
      # both are ahead the sat_pos, take the closest
      if (t1>=0) and (t2>=0) :
        intersection = sat_pos + min(t1,t2)*sat_los
        bin = 1
        if np.linalg.norm(intersection-sat_pos) < 1e-5:
          bin = 2
        return intersection, bin
      # both are behinf the sat_pos, take none
      if (t1 <0) and (t2<0):
        intersection = np.array([0,0,0])
        bin = -1
        return intersection, bin
      # One is ahead and the other behind, take the positive one
      intersection = sat_pos + max(t1,t2)*sat_los
      bin = 1
      if np.linalg.norm(intersection-sat_pos)<1e-5:
        bin = 2
      return intersection, bin
    
    # One intersection in right region, second one no
    if (intersection1[2]>0) and (intersection2[2]<0):
      # intersection is ahead
      if (t1>=0):
        intersection = intersection1
        bin = 1
        if np.linalg.norm(intersection-sat_pos) < 1e-5:
          bin = 2
        return intersection, bin
      else: # intersection is behind
        intersection = np.array([0,0,0])
        bin = -1
        return intersection, bin
    
    if (intersection1[2]<0) and (intersection2[2]>0):
      # intersection is ahead
      if (t2 >= 0):
        intersection = intersection2
        bin = 1
        if np.linalg.norm(intersection-sat_pos)<1e-5:
          bin = 2
        return intersection, bin
      else: # intersection is behind
        intersection = [0,0,0]
        bin = -1
        return intersection, bin

    if (intersection1[2]<0 and intersection2[2]<0):
      intersection = np.array([0,0,0])
      bin = -1
      return intersection, bin

  if (th > 90):
    intersection1 = sat_pos + t1*sat_los
    intersection2 = sat_pos + t2*sat_los

    # both are below the ecliptic plane
    if (intersection1[2]<0) and (intersection2[2]<0):
      # both are ahead  sat_pos, take the closest one
      if (t1>=0) and (t2>=0):
        intersection = sat_pos + min(t1,t2)*sat_los
        bin = 1
        if np.linalg.norm(intersection-sat_pos) < 1e-5:
          bin = 2
        return intersection, bin
      
      # both are behind  sat_pos, take none
      if (t1<0) and (t2<0):
        intersection = np.array([0,0,0])
        bin = -1
        return intersection, bin
      
      # one is ahead and the oher is behind, take the positive one
      intersection = sat_pos + max(t1,t2)*sat_los
      bin = 1
      if np.linalg.norm(intersection-sat_pos) < 1e-5:
        bin = 2
      return intersection, bin
    
    # One intersection in right region, second one no
    if (intersection1[2]<0) and (intersection2[2]>0):
      # intersection is ahead
      if (t1>=0):
        intersection = intersection1
        bin = 1
        if np.linalg.norm(intersection-sat_pos) < 1e-5:
          bin = 2
        return intersection, bin
      else: # intersection is behind
        intersection = np.array([0,0,0])
        bin = -1
        return intersection, bin
    
    if (intersection1[2]>0) and (intersection2[2]<0):
      #intersection is ahead
      if (t2>=0):
        intersection = intersection2
        bin = 1
        if np.linalg.norm(intersection-sat_pos) < 1e-5:
          bin = 2
        return intersection, bin
      else: # intersection is behind
        intersection = np.array([0,0,0])
        bin = -1
        return intersection, bin
      
    if (intersection1[2]>0) and (intersection1[2]>0):
      intersection = np.array([0,0,0])
      bin = -1
      return intersection, bin

#-------------------------------------------------------------------------------
def line_cylinder_intersection(line_or,line_uv,radius,plane):
  # Intersection between a line and a cylinder that lies in an axis.
  # Inputs :
  #   line_or = is a vector containing (X,Y,Z) values in GSE coordinate system.
  #   line_uv = is a unit vector (X,Y,Z) in GSE, "LOS"
  #   radius  = is the radius of the cylinder
  #   plane   = the plane of the locations of the circunference section of the cylinder
  # Outputs:
  #    intercep = X,Y,Z of the intersection (why "c"ep ? IDK)
  #    bin = if -1 then there is not intersection.
  #  HINT: For SHADOW (NIGHT SIDE) Intersection use 'YZ' and compare if the X value of
  #  intercep is negative

  line_uv = line_uv/np.linalg.norm(line_uv)

  if (plane == 'XY'):
    temp1 = line_or[0]
    temp2 = line_or[1]
    temp1d = line_uv[0]
    temp2d = line_uv[1]

  if (plane == 'YZ'):
    temp1 = line_or[2]
    temp2 = line_or[1]
    temp1d = line_uv[2]
    temp2d = line_uv[1]

  if (plane == 'XZ'):
    temp1 = line_or[0]
    temp2 = line_or[2]
    temp1d = line_uv[0]
    temp2d = line_uv[2]   

  a = temp1d**2 + temp2d**2
  b = 2*temp1*temp1d + 2*temp2*temp2d
  c = temp1**2 + temp2**2 - radius**2
  b24ac = b**2 - 4*a*c

  if (b24ac<0):
    bin = -1
    interception = np.array([0,0,0])
    return interception, bin

  t0 = (-b+np.sqrt(b24ac))/(2*a)
  t1 = (-b-np.sqrt(b24ac))/(2*a)

  intercep0 = line_or + t0*line_uv
  intercep1 = line_or + t1*line_uv

  if (np.linalg.norm(intercep0-line_or)<np.linalg.norm(intercep1-line_or)):
    intersection = intercep0
    bin = 1
  else:
    intersection = intercep1
    bin = 1
  
  return intersection,bin

#-------------------------------------------------------------------------------
def generateLOSfromImager(fov,pixangres,target_los):
  # target_los should be a unit vector
  target_los = target_los/np.linalg.norm(target_los)
  # Initial directions for imager's LOSs
  ori_dir = np.array([1,0,0])
  numpix  = int(np.ceil(fov/pixangres))
  # initial LOS (pointing to 1,0,0)
  los     = np.zeros((numpix*numpix,3))
  # rotated LOS (pointing to target)
  rlos    = np.zeros((numpix*numpix,3))
  r       = 1 # unit vector
  clos    = 0 # just a counter 
  #print(target_los)

  for k1 in range(numpix):
    for k2 in range(numpix):
      theta     = (np.pi/180)*fov/2 - k1*(np.pi/180)*pixangres
      phi       = (np.pi/180)*fov/2 - k2*(np.pi/180)*pixangres
      [x,y,z]   = sph2cart(phi,theta,r) 
      los[clos,:] = [x,y,z]
      clos      = clos + 1 
  #print(target_los)
  if (target_los[0]==-1) and (target_los[1]==0) and (target_los[2]==0):
    # Rotate 180 degrees
    rlos = los
    for k3 in range(len(los)):
      rlos[k3,0] = -rlos[k3,0]

    return rlos, numpix

  else:
    # Rodriguez Formula to rotate LOS towards DESIRED_LOS
    V   = np.cross(ori_dir,target_los);
    SSC = np.array([[0,-V[2],V[1]],[ V[2],0,-V[0]],[ -V[1],V[0],0]]);
    R   = np.eye(3) + SSC + (np.dot(SSC,SSC))*(1-np.dot(ori_dir,target_los))/(np.linalg.norm(V)**2);

    for k3 in range(len(los)):
      rlos[k3,:] = np.dot(R,los[k3,:])

    return rlos, numpix

#-------------------------------------------------------------------------------
def los_voxel_intersection(pos,los,voxelid,exosgrid):
  rindex = voxelid[0]
  pindex = voxelid[1]
  tindex = voxelid[2]

  # intersection with six planes within the voxel
  r1 = exosgrid.rvals[rindex]
  r2 = exosgrid.rvals[rindex+1]
  th1 = exosgrid.tvals[tindex]
  th2 = exosgrid.tvals[tindex+1]
  phi1 = exosgrid.pvals[pindex]
  phi2 = exosgrid.pvals[pindex+1]

  # plane information
  [x0,y0,z0] = sph2cart(phi1*np.pi/180,np.pi/2-th1*np.pi/180,r1)
  [x1,y1,z1] = sph2cart(phi2*np.pi/180,np.pi/2-th1*np.pi/180,r1)
  [x2,y2,z2] = sph2cart(phi1*np.pi/180,np.pi/2-th2*np.pi/180,r1)
  [x3,y3,z3] = sph2cart(phi2*np.pi/180,np.pi/2-th2*np.pi/180,r1)

  # other planes
  planeright = np.array([[x1,y1,z1],[x3,y3,z3],[0,0,0]])
  planeleft  = np.array([[x0,y0,z0],[x2,y2,z2],[0,0,0]])

  # indicator
  insd  = 1

  while True:
    # Looking for intersection line and sphere -> R1
    [cross,bin] = line_sph_intersection(pos,los,r1)
    if bin == 1:
      [phi,theta,r] = cart2sph(cross[0],cross[1],cross[2])
      phi = phi*180/np.pi
      if phi < 0:
        phi = phi + 360
      colat = 90-theta*180/np.pi
      if (colat>=th1) and (colat<th2) and (phi>=phi1) and (phi<phi2):
        rindex = rindex - 1
        if (rindex<0):
          insd = 0
        break

    # Looking for intersection line and sphere -> R2
    [cross,bin] = line_sph_intersection(pos,los,r2)
    if bin == 1:
      [phi,theta,r] = cart2sph(cross[0],cross[1],cross[2])
      phi = phi*180/np.pi
      if phi < 0:
        phi = phi + 360
      colat = 90-theta*180/np.pi
      if (colat>=th1) and (colat<th2) and (phi>=phi1) and (phi<phi2):
        rindex = rindex + 1
        if (rindex >= exosgrid.numR):
          insd = 0
        break

    # Looking for intersection line and upper plane
    [cross,bin] = line_cone_intersection(pos,los,th1)
    if bin == 1:
      [phi,theta,r] = cart2sph(cross[0],cross[1],cross[2])
      phi = phi*180/np.pi
      if phi < 0:
        phi = phi + 360
      colat = 90-theta*180/np.pi
      if (phi>=phi1) and (phi<phi2) and (r>=r1) and (r<r2):
        tindex = tindex - 1
        if tindex<0:
          insd = 0
        break

    # Looking for intersection line and lower plane
    [cross,bin] = line_cone_intersection(pos,los,th2)
    if bin == 1:
      [phi,theta,r] = cart2sph(cross[0],cross[1],cross[2])
      phi = phi*180/np.pi
      if phi < 0:
        phi = phi + 360
      colat = 90-theta*180/np.pi
      if (phi>=phi1) and (phi<phi2) and (r>=r1) and (r<r2):
        tindex = tindex + 1
        if tindex>=exosgrid.numT:
          insd = 0
        break

    # Looking for intersection line and left plane
    [cross,bin] = line_plane_intersection(pos,los,planeleft)
    if bin == 1:
      [phi,theta,r] = cart2sph(cross[0],cross[1],cross[2])
      phi = phi*180/np.pi
      if phi < 0:
        phi = phi + 360
      colat = 90-theta*180/np.pi
      if (colat>=th1) and (colat<th2) and (r>=r1) and (r<r2):
        pindex = pindex - 1
        if pindex<0:
          pindex = exosgrid.numP - 1. # verify
        break

    # Looking for intersection line and left plane
    [cross,bin] = line_plane_intersection(pos,los,planeright)
    if bin == 1:
      [phi,theta,r] = cart2sph(cross[0],cross[1],cross[2])
      phi = phi*180/np.pi
      if phi < 0:
        phi = phi + 360
      colat = 90-theta*180/np.pi
      if (colat>=th1) and (colat<th2) and (r>=r1) and (r<r2):
        pindex = pindex + 1
        if pindex>=exosgrid.numP:
          pindex = 0 # verify
        break
      
    insd = 2;
    sectorlength    = [];
    newpos          = [];
    newvoxelid      = [];
    return newvoxelid, sectorlength, newpos, insd

  sectorlength    = np.linalg.norm(pos - cross);
  newpos          = cross;
  newvoxelid      = np.array([int(rindex),int(pindex),int(tindex)])
  return newvoxelid, sectorlength, newpos, insd

#-------------------------------------------------------------------------------
def getvoxelID(pos,exosgrid):
  # Converting to colatitude, phi and radius
  [phi,theta,r] = cart2sph(pos[0],pos[1],pos[2])
  #print(r)
  phi = phi*180/np.pi
  if phi<0:
    phi = phi + 360
  colat = 90 - theta*180/np.pi

  # Identifying voxelID
  ### tvals
  temp1 = np.abs(colat-exosgrid.tvals)
  temp2 = np.sign(colat-exosgrid.tvals)
  amin  = np.argmin(temp1)

  if temp2[amin] >=0:
    tindex = amin
  else:
    tindex = amin-1
  
  ### pvals
  temp1 = np.abs(phi-exosgrid.pvals)
  temp2 = np.sign(phi-exosgrid.pvals)
  amin  = np.argmin(temp1)

  if temp2[amin]>=0:
    pindex = amin
  else:
    pindex = amin-1

  ### rvals
  temp1 = np.abs(r-exosgrid.rvals)
  temp2 = np.sign(r-exosgrid.rvals)
  amin  = np.argmin(temp1)

  if temp2[amin]>=0:
    rindex = amin
  else:
    rindex = amin-1
  
  if rindex == exosgrid.rvals.shape[0]-1:
    rindex = rindex - 1;

  voxelID = [rindex,pindex,tindex]

  return voxelID

#-------------------------------------------------------------------------------
def get_Lpartial(sat_pos,sat_los,exosgrid):
  asunward_d  = np.array([-1,0,0]) #antisunward direction
  sat_los      = sat_los/np.linalg.norm(sat_los)
  radius_sat  = np.linalg.norm(sat_pos)
  angle_I     = np.arccos(sat_los.dot(asunward_d))*180/np.pi
  geodistlos  = np.linalg.norm(sat_pos - sat_los.dot(sat_pos)*sat_los)
  #print(radius_sat)

  if (geodistlos<=exosgrid.rmax) and (geodistlos>=exosgrid.rmin):
    if (radius_sat>exosgrid.rmax): #outside the solution domain
      [current_pos,bin] = line_sph_intersection(sat_pos,sat_los,exosgrid.rmax)
      if bin!=1:
        lpartial = np.array([[-1]])
        angle_I  = np.array([[0]])
        print('test0')
        return lpartial,angle_I
    else:
      current_pos = sat_pos
  else:
    lpartial = np.array([[-1]])
    angle_I  = np.array([[0]])
    print('test1')
    print(geodistlos)
    return lpartial,angle_I

  [intersection, bin] = line_cylinder_intersection(sat_pos,sat_los,3,'YZ')
  if (bin!=-1):
    if (intersection[0]<0) and (intersection[2]>-30):
      lpartial = np.array([[-1]])
      angle_I  = np.array([[0]])
      print('test2')
      return lpartial, angle_I
  
  voxelID = getvoxelID(current_pos,exosgrid)
  if (voxelID[0]==exosgrid.numR) or (voxelID[1]==exosgrid.numP) or (voxelID[2]==exosgrid.numT):
    lpartial = np.array([[-1]])
    angle_I  = np.array([[0]])
    print('\ntest3')
    #print(current_pos)
    #print(voxelID)
    return lpartial, angle_I

  # LPARTIAL    = sparse(1,TOMO.NUMVOXELS);
  lpartial = csr_matrix((1,int(exosgrid.numvoxels)))

  while True:
    [newvoxelid,sectorlength,newpos,insd]= los_voxel_intersection(current_pos,sat_los,voxelID,exosgrid) # verify this function!  

    if (insd==0): # outside the grid
      #save sectorlength and voxelid
      index = voxelID[0] + voxelID[1]*exosgrid.numR + voxelID[2]*exosgrid.numR*exosgrid.numP # verify "voxelID[1]-1"
      #LPARTIAL(1,index) = LPARTIAL(1,index) + sectorlength;
      lpartial[0,int(index)] = lpartial[0,int(index)]  + sectorlength
      #print('insd=0')
      break
    
    if (insd==2): # wrong, dont save
      lpartial = np.array([[-1]])
      angle_I  = np.array([[0]])
      #print('insd=2')
      return lpartial, angle_I
    
    # save sectorlength and voxelid
    index = voxelID[0] + voxelID[1]*exosgrid.numR + voxelID[2]*exosgrid.numR*exosgrid.numP # verify "voxelID[1]-1"
    #LPARTIAL(1,index) = LPARTIAL(1,index) + sectorlength;
    lpartial[0,int(index)] = lpartial[0,int(index)]  + sectorlength

    # Update CURRENTPOS and VOXELID
    current_pos   = newpos;
    voxelID       = newvoxelid;

  return lpartial, angle_I

#-------------------------------------------------------------------------------
def generateObservationMatrix(los,pos,exosgrid):
  # Generate the Observation Matrix
  ObsMatrix_t = csr_matrix((los.shape[0],int(exosgrid.numvoxels)))
  # Generate Vector of scattering angles
  AngleI_t   = np.zeros((los.shape[0],1))

  #counter = 0
  for i in tqdm(range(los.shape[0]),"Processing...",ascii=False, ncols=75):
    #print(i)
    [lpartial,aI] = get_Lpartial(pos[i,:],los[i,:],exosgrid)
    if (lpartial.shape[1] > 1):
      ObsMatrix_t[i,:] = lpartial
      AngleI_t[i] = aI
      #counter = counter + 1

  # Generating final ObsMatrix and AngleI variables
  #ObsMatrix = csr_matrix((counter,int(exosgrid.numvoxels)))
  #AngleI    = np.zeros((counter,1))

  # Copying data to final variables
  #ObsMatrix = ObsMatrix_t[0:counter,:]
  #AngleI    = AngleI_t[0:counter,:]

  return ObsMatrix_t,AngleI_t

#-------------------------------------------------------------------------------
#def generate3DHmodel(model,exosgrid):
#  H = np.zeros((int(exosgrid.numvoxels),1))
#  theta = np.array([[0.0]])
#  rad = np.array([[0.0]])
#  phi = np.array([[0.0]])

#  for t_id in range(int(exosgrid.numT)):
#    theta[0,0] = exosgrid.tvals[t_id]+exosgrid.tstep/2
#    theta[0,0] = theta[0,0]*np.pi/180
#    for p_id in range(int(exosgrid.numP)):
#      phi[0,0] = exosgrid.pvals[p_id]+exosgrid.pstep/2
#      phi[0,0] = phi[0,0]*np.pi/180
#      for r_id in range(int(exosgrid.numR)):
#        rad[0,0] = exosgrid.rvals[r_id]+exosgrid.rstep/2
#        H[int(r_id+p_id*exosgrid.numR+t_id*exosgrid.numR*exosgrid.numP),0] = get_density(model,rad,theta,phi)
#        #print(rad,theta,phi)

#  return H
def generate3DHmodel(model, exosgrid):
    H = np.zeros((int(exosgrid.numvoxels), 1), dtype=float)
    theta = np.array([[0.0]])
    rad   = np.array([[0.0]])
    phi   = np.array([[0.0]])

    for t_id in range(int(exosgrid.numT)):
        theta[0,0] = exosgrid.tvals[t_id] + exosgrid.tstep/2
        theta[0,0] = theta[0,0] * np.pi/180.0

        for p_id in range(int(exosgrid.numP)):
            phi[0,0] = exosgrid.pvals[p_id] + exosgrid.pstep/2
            phi[0,0] = phi[0,0] * np.pi/180.0

            for r_id in range(int(exosgrid.numR)):
                rad[0,0] = exosgrid.rvals[r_id] + exosgrid.rstep/2

                idx = int(r_id + p_id*exosgrid.numR + t_id*exosgrid.numR*exosgrid.numP)
                H[idx,0] = get_density(model, rad, theta, phi)
  return H


#-------------------------------------------------------------------------------
def getHolstein(tau):
  temp = quad(noIntegratedT,0,5,args=tau)
  return (2.0/np.sqrt(np.pi))*temp[0]

#-------------------------------------------------------------------------------
def noIntegratedT(x,tau):
  return np.exp(-x**2)*np.exp(-tau*np.exp(-x**2))

#-------------------------------------------------------------------------------
def generateIntensityOpticallyThin(irradiance,r_los,r_pos,model,dl = 0.1,maxRAD = 8, minRAD = 3):
  
  lyman_alpha = 121.6e-9 # m
  lightspeed  = 3e8 # m/s
  planck      = 6.63e-34 # J.s
  f_flux      = (irradiance*lyman_alpha)*(1e-4)/(planck*lightspeed) #ph/s/m2
  g_factor    = 3.47e-4*(f_flux/1e11)**(1.21) # 1/s

  print('g factor used in this analysis = ',g_factor)
  
  
  Xdir      = np.array([1,0,0])
  # vector for output
  Intensity_v = np.zeros((len(r_los),1))
  # temporal variable
  intensity = 0#np.zeros((len(r_los),1))

  # Main loop along the LOSs
  for i in tqdm(range(len(r_los)),"Processing...",ascii=False, ncols = 75):
    pf_ang  = np.arccos(Xdir.dot(r_los[i,:])/(np.linalg.norm(Xdir)*np.linalg.norm(r_los[i,:])))
    pf      = (11./12.) + ((1./4.) * 0.5 * (np.cos(2*pf_ang) +1))
    radius  = np.sqrt(r_pos[i,0]**2 + r_pos[i,1]**2 + r_pos[i,2]**2)  # in geocentric RE

    if (radius>maxRAD): #satellite outside the solution domain
      [current_pos,bin] = line_sph_intersection(r_pos[i,:],r_los[i,:],maxRAD)
      if bin!=1:
        print('error')
    else:
      current_pos = r_pos[i,:]

    intensity = 0
    while True:
      #radius      = np.sqrt(current_pos[0]**2 + current_pos[1]**2 + current_pos[2]**2)  # in geocentric RE
      [phi,theta,r] = cart2sph(current_pos[0],current_pos[1],current_pos[2])
      phi = phi*180.0/np.pi
      if phi < 0:
        phi = phi + 360.0
      colat = 90.0-theta*180.0/np.pi

      if (r>maxRAD) or (r < minRAD):
        #print(radius)
        break
      # Calculate intensity
      intensity = intensity + (pf*g_factor/10.0**6)*get_density(model,r,colat*np.pi/180,phi*np.pi/180)*dl*(6371*10**5)
      # Update the current position
      current_pos = current_pos + dl*r_los[i,:] #in RE

    # Save Intensity
    Intensity_v[i,0] = intensity
  
  return Intensity_v

#-------------------------------------------------------------------------------
# NEW CODE TO HANDLE OF ADDITIONAL MODELS 

# ----------------------------
# Constants / helpers
# ----------------------------
RE_KM = 6371.0

def _to_numpy(x):
    return np.asarray(x, dtype=float)

def _deg2rad_if_needed(angle, degrees):
    a = _to_numpy(angle)
    return np.deg2rad(a) if degrees else a

def _safe_sqrt(x):
    return np.sqrt(np.clip(x, 0.0, None))

def _factorial(n):
    return math.factorial(int(n))

def _norm_lm(l, m):
    # Real SH "Legendre polynomial" normalization (common in geophysics):
    # Y_lm(theta) = N_lm * P_l^m(cos(theta)), with N_lm = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)
    return math.sqrt((2*l + 1)/(4*math.pi) * _factorial(l-m)/_factorial(l+m))

def _assoc_legendre_lm(l, m, x):
    """
    Associated Legendre P_l^m(x) for l<=3 (hardcoded, stable, no scipy).
    Uses the Condon-Shortley phase (-1)^m (included in formulas below).
    x can be numpy array.
    """
    x = _to_numpy(x)
    s = _safe_sqrt(1.0 - x*x)

    if l == 0 and m == 0:
        return np.ones_like(x)

    if l == 1:
        if m == 0:
            return x
        if m == 1:
            return -s

    if l == 2:
        if m == 0:
            return 0.5*(3.0*x*x - 1.0)
        if m == 1:
            return -3.0*x*s
        if m == 2:
            return 3.0*(1.0 - x*x)

    if l == 3:
        if m == 0:
            return 0.5*(5.0*x**3 - 3.0*x)
        if m == 1:
            return -(3.0/2.0)*(5.0*x*x - 1.0)*s
        if m == 2:
            return 15.0*x*(1.0 - x*x)
        if m == 3:
            return -15.0*(s**3)

    raise ValueError(f"assoc_legendre only implemented for l<=3, got l={l}, m={m}")

def ylm_theta(l, m, theta):
    """
    Y_lm(theta) used by these SHR models (no phi dependence here).
    theta is colatitude (0 at +Z/North).
    """
    theta = _to_numpy(theta)
    x = np.cos(theta)
    P = _assoc_legendre_lm(l, m, x)
    return _norm_lm(l, m) * P

def _shr_real(theta, phi, coeffs, lmax):
    """
    SHR = sqrt(4π) * Σ_{l=0..lmax} Σ_{m=0..l} [A_lm cos(mφ) + B_lm sin(mφ)] Y_lm(θ)
    coeffs dict must provide A[(l,m)] and B[(l,m)] (missing -> 0), plus A00=1 typically.
    """
    theta = _to_numpy(theta)
    phi   = _to_numpy(phi)

    out = np.zeros(np.broadcast(theta, phi).shape, dtype=float)
    for l in range(lmax + 1):
        for m in range(l + 1):
            Y = ylm_theta(l, m, theta)
            A = coeffs.get(("A", l, m), 0.0)
            B = coeffs.get(("B", l, m), 0.0)
            if m == 0:
                out = out + A * Y
            else:
                out = out + (A*np.cos(m*phi) + B*np.sin(m*phi)) * Y
    return math.sqrt(4.0*math.pi) * out

# ----------------------------
# Model coefficient builders
# ----------------------------
def _zoennchen2015_coeffs(which):
    """
    Zoennchen et al. (2015): lmax=2, A00=1, Alm & Blm = (const + const*ln(r)) * 1e-4 (except A00).
    'which' in {'min', 'max'}
    """
    if which == "min":
        # Table 1 (solar minimum 2008/2010) :contentReference[oaicite:7]{index=7}
        c = 12264.1
        k = 2.87646
        a = { (1,0):(  938.00,  135.41),
              (1,1):(   92.20,  198.16),
              (2,0):( -385.26, -597.06),
              (2,1):( 2042.26, -916.65),
              (2,2):( -421.34, 1196.10) }
        b = { (1,1):( 4870.41, -2632.88),
              (2,1):(-2506.10,  1578.28),
              (2,2):( 2783.95, -1331.32) }
    elif which == "max":
        # Table 2 (near-solar-maximum 2012) :contentReference[oaicite:8]{index=8}
        c = 16840.9
        k = 2.74640
        a = { (1,0):( -921.29,   790.11),
              (1,1):( 6763.12, -3088.94),
              (2,0):( -494.96,  -405.36),
              (2,1):( -284.02,    44.03),
              (2,2):( -556.96,  1303.13) }
        b = { (1,1):( 1289.94,  -788.54),
              (2,1):( -753.84,   256.56),
              (2,2):( 2029.43, -1084.30) }
    else:
        raise ValueError("Zoennchen2015 which must be 'min' or 'max'")

    def coeffs_at_r(r_re):
        r_re = _to_numpy(r_re)
        lr = np.log(r_re)
        coeffs = {("A",0,0): 1.0}
        scale = 1e-4
        # A terms
        for (l,m),(aa,bb) in a.items():
            coeffs[("A",l,m)] = (aa + bb*lr)*scale
        # B terms
        for (l,m),(pp,qq) in b.items():
            coeffs[("B",l,m)] = (pp + qq*lr)*scale
        # missing terms assumed 0
        return coeffs

    meta = {"type":"zoennchen2015", "which":which, "c":c, "k":k, "lmax":2}
    return meta, coeffs_at_r

def _zoennchen2024_coeffs(which):
    """
    Zoennchen et al. (2024): lmax=3, Eq(5) includes d^(1/r) term. :contentReference[oaicite:9]{index=9}
    Coeffs from Table 3 for 2008/2013/2015. :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
    """
    if which == "2008":
        c,k,d = 4400.47602, 2.35863972, 5.13191135
        A = {
            (1,0):( 0.10378384,      -0.0549490022),
            (1,1):( 0.0209286265,     0.00027287255),
            (2,0):( 0.0973649071,    -0.119977357),
            (2,1):(-0.145366729,      0.0937117025),
            (2,2):(-0.277012528,      0.276749422),
            (3,0):(-1.07414979e-5,   -0.0110100711),
            (3,1):( 0.000107379013,  -1.78917719e-6),
            (3,2):( 2.64592101e-5,    0.000878167056),
            (3,3):(-0.189335716,      0.132502861),
        }
        B = {
            (1,1):(-0.0106621543,     0.0221945739),
            (2,1):(-0.0900472936,     0.0580710504),
            (2,2):( 0.163972774,     -0.0873760161),
            (3,1):(-0.0851136046,     0.0684112054),
            (3,2):( 0.347028204,     -0.228867241),
            (3,3):(-3.63528687e-6,   -0.00871796055),
        }
    elif which == "2013":
        c,k,d = 8143.32369, 2.46136837, 1.91745002
        A = {
            (1,0):( 1.88713836e-7,   -0.0101115442),
            (1,1):(-0.000372983798,  -3.72355169e-5),
            (2,0):(-0.000925215858, -0.0134752631),
            (2,1):(-0.0189771246,    0.0196004454),
            (2,2):(-0.0851061204,    0.106488415),
            (3,0):(-0.0162650162,    0.00690910979),
            (3,1):( 0.0047678757,   -9.00328944e-6),
            (3,2):(-9.76424742e-6,   0.00573736914),
            (3,3):(-4.02268369e-6,   0.00194601785),
        }
        B = {
            (1,1):(-0.0442302801,     0.0208651138),
            (2,1):(-0.0649227416,     0.0324600394),
            (2,2):(-0.000513397304,   6.85089551e-5),
            (3,1):(-0.0907362653,     0.0656108159),
            (3,2):( 1.56635585e-5,   -0.00832095984),
            (3,3):( 0.000252872776,  -0.0067339901),
        }
    elif which == "2015":
        c,k,d = 8022.87883, 2.46826669, 2.40919011
        A = {
            (1,0):(-0.00115599731,   -8.55009777e-5),
            (1,1):(-0.0321731569,     6.44984436e-5),
            (2,0):( 0.0528720519,    -0.0483273322),
            (2,1):(-0.135265871,      0.0974773937),
            (2,2):( 0.0640442767,    -3.19342462e-5),
            (3,0):( 0.0511557829,    -0.0417547088),
            (3,1):(-0.183336474,      0.134325125),
            (3,2):( 0.014359638,     -8.82895819e-6),
            (3,3):( 3.78566663e-6,   -0.0004674004),
        }
        B = {
            (1,1):(-0.0169112053,    -0.0157071887),
            (2,1):( 6.48856934e-6,    0.000228402417),
            (2,2):(-0.0734408996,     0.0421809581),
            (3,1):(-0.0071425359,     0.0151312125),
            (3,2):(-1.10036313e-6,   -0.000117698068),
            (3,3):(-2.41106162e-5,   -0.0088006464),
        }
    else:
        raise ValueError("Zoennchen2024 which must be '2008', '2013', or '2015'")

    def coeffs_at_r(r_re, r_limit=6.0):
        # Freeze angular coefficients for r>6 Re (use r_eff for ln(r) in A/B),
        # but radial N(r) uses the real r. :contentReference[oaicite:13]{index=13}
        r_re = _to_numpy(r_re)
        r_eff = np.minimum(r_re, r_limit)
        lr = np.log(r_eff)

        coeffs = {("A",0,0): 1.0}
        for (l,m),(aa,bb) in A.items():
            coeffs[("A",l,m)] = aa + bb*lr
        for (l,m),(pp,qq) in B.items():
            coeffs[("B",l,m)] = pp + qq*lr
        # B10=B20=B30=0 in the table :contentReference[oaicite:14]{index=14}
        coeffs[("B",1,0)] = 0.0
        coeffs[("B",2,0)] = 0.0
        coeffs[("B",3,0)] = 0.0
        return coeffs

    meta = {"type":"zoennchen2024", "which":which, "c":c, "k":k, "d":d, "lmax":3}
    return meta, coeffs_at_r

def _bailey2008_coeffs():
    """
    Bailey & Gruntman (TWINS 11-Jun-2008): coefficients table provided (r in km, n in cm^-3). :contentReference[oaicite:15]{index=15}
    We use:
      N(r_km) = p * r_km^k
      A_lm(r_km) = a_lm + b_lm * r_km
      B_lm(r_km) = c_lm + d_lm * r_km
    with lmax=2, A00=1.
    """
    # Table values :contentReference[oaicite:16]{index=16}
    p = 4.1118e13
    k = -2.5446

    A_lin = {
        (1,0):(-4.8992e-2, -1.8720e-6),
        (1,1):(-3.8248e-1,  9.0636e-6),
        (2,0):( 1.5739e-1, -6.1959e-6),
        (2,1):(-6.9198e-2,  4.5477e-6),
        (2,2):(-1.0148e-1,  1.4873e-6),
    }
    B_lin = {
        (1,1):(-4.8547e-2, -2.1587e-6),
        (2,1):( 2.1922e-1, -7.0881e-6),
        (2,2):(-8.8242e-2,  4.2384e-6),
    }

    def coeffs_at_r(r_km):
        r_km = _to_numpy(r_km)
        coeffs = {("A",0,0): 1.0}
        for (l,m),(aa,bb) in A_lin.items():
            coeffs[("A",l,m)] = aa + bb*r_km
        for (l,m),(cc,dd) in B_lin.items():
            coeffs[("B",l,m)] = cc + dd*r_km
        return coeffs

    meta = {"type":"bailey2008", "p":p, "k":k, "lmax":2}
    return meta, coeffs_at_r

# ----------------------------
# Public API
# ----------------------------
def available_h_models():
    return [
        "bailey_2008",
        "zoennchen_2015_min",
        "zoennchen_2015_max",
        "zoennchen_2024_2008",
        "zoennchen_2024_2013",
        "zoennchen_2024_2015",
    ]

def h_model_meta(model_name):
    model_name = str(model_name).lower().strip()
    if model_name == "bailey_2008":
        meta, _ = _bailey2008_coeffs()
        return meta
    if model_name == "zoennchen_2015_min":
        meta, _ = _zoennchen2015_coeffs("min")
        return meta
    if model_name == "zoennchen_2015_max":
        meta, _ = _zoennchen2015_coeffs("max")
        return meta
    if model_name == "zoennchen_2024_2008":
        meta, _ = _zoennchen2024_coeffs("2008")
        return meta
    if model_name == "zoennchen_2024_2013":
        meta, _ = _zoennchen2024_coeffs("2013")
        return meta
    if model_name == "zoennchen_2024_2015":
        meta, _ = _zoennchen2024_coeffs("2015")
        return meta
    raise ValueError(f"Unknown model_name='{model_name}'. Use available_h_models().")

def h_density(model_name, r, theta, phi, *, degrees=False, r_units="Re", r_limit_zoennchen2024=6.0):
    """
    Compute nH for any (r, theta, phi).

    Parameters
    ----------
    model_name : str
        One of available_h_models()
    r : float or array
        Radius. If r_units="Re": in Earth radii. If "km": in km.
    theta : float or array
        Colatitude (0 at +Z / north pole).
    phi : float or array
        Longitude (radians by default). If degrees=True, theta & phi in degrees.
    degrees : bool
        If True, interprets theta & phi in degrees.
    r_units : {"Re","km"}
        Units for r input.
    r_limit_zoennchen2024 : float
        Freeze angular coefficients at r_eff=min(r, r_limit) for Zoennchen2024. :contentReference[oaicite:17]{index=17}

    Returns
    -------
    nH : float or numpy array
        Density in cm^-3.
    """
    model_name = str(model_name).lower().strip()
    theta = _deg2rad_if_needed(theta, degrees)
    phi   = _deg2rad_if_needed(phi, degrees)

    r_in = _to_numpy(r)
    if r_units.lower() == "re":
        r_re = r_in
        r_km = r_in * RE_KM
    elif r_units.lower() == "km":
        r_km = r_in
        r_re = r_in / RE_KM
    else:
        raise ValueError("r_units must be 'Re' or 'km'")

    if model_name == "bailey_2008":
        meta, coeffs_at_r = _bailey2008_coeffs()
        # Bailey uses r in km for N(r)=p*r^k :contentReference[oaicite:18]{index=18}
        N = meta["p"] * (r_km ** meta["k"])
        coeffs = coeffs_at_r(r_km)
        SHR = _shr_real(theta, phi, coeffs, meta["lmax"])
        return N * SHR

    if model_name in ("zoennchen_2015_min", "zoennchen_2015_max"):
        which = "min" if model_name.endswith("_min") else "max"
        meta, coeffs_at_r = _zoennchen2015_coeffs(which)
        N = meta["c"] * (r_re ** (-meta["k"]))
        coeffs = coeffs_at_r(r_re)
        SHR = _shr_real(theta, phi, coeffs, meta["lmax"])
        return N * SHR

    if model_name.startswith("zoennchen_2024_"):
        which = model_name.split("_")[-1]  # 2008/2013/2015
        meta, coeffs_at_r = _zoennchen2024_coeffs(which)
        # Eq(5): c*r^{-k}*d^{1/r}*SHR  :contentReference[oaicite:19]{index=19}
        N = meta["c"] * (r_re ** (-meta["k"])) * (meta["d"] ** (1.0 / r_re))
        coeffs = coeffs_at_r(r_re, r_limit=r_limit_zoennchen2024)
        SHR = _shr_real(theta, phi, coeffs, meta["lmax"])
        return N * SHR

    raise ValueError(f"Unknown model_name='{model_name}'. Use available_h_models().")

# Convenience wrappers (optional)
def h_density_bailey2008(r, theta, phi, *, degrees=False, r_units="Re"):
    return h_density("bailey_2008", r, theta, phi, degrees=degrees, r_units=r_units)

def h_density_zoennchen2015_min(r, theta, phi, *, degrees=False, r_units="Re"):
    return h_density("zoennchen_2015_min", r, theta, phi, degrees=degrees, r_units=r_units)

def h_density_zoennchen2015_max(r, theta, phi, *, degrees=False, r_units="Re"):
    return h_density("zoennchen_2015_max", r, theta, phi, degrees=degrees, r_units=r_units)

def h_density_zoennchen2024_2008(r, theta, phi, *, degrees=False, r_units="Re", r_limit=6.0):
    return h_density("zoennchen_2024_2008", r, theta, phi, degrees=degrees, r_units=r_units, r_limit_zoennchen2024=r_limit)

def h_density_zoennchen2024_2013(r, theta, phi, *, degrees=False, r_units="Re", r_limit=6.0):
    return h_density("zoennchen_2024_2013", r, theta, phi, degrees=degrees, r_units=r_units, r_limit_zoennchen2024=r_limit)

def h_density_zoennchen2024_2015(r, theta, phi, *, degrees=False, r_units="Re", r_limit=6.0):
    return h_density("zoennchen_2024_2015", r, theta, phi, degrees=degrees, r_units=r_units, r_limit_zoennchen2024=r_limit)



#-------------------------------------------------------------------------------
#def draw3DHmodel(model,exosgrid,plane,arg,plotb):
#  H = generate3DHmodel(model,exosgrid)
#  H = np.reshape(H,(int(exosgrid.numT), int(exosgrid.numP), int(exosgrid.numR)))

#  if plane == 'map':
#    # verifying that arg should be between exosgrid.rmin and exosgrid.rmax
#    if (arg<exosgrid.rmin) or (arg>exosgrid.rmax):
#      print('Radius outside the valid limits')
#      return -1
#    temp1 = abs(exosgrid.rvals - arg)
#    temp2 = np.argmin(temp1)
#    toPlot = H[:,:,int(temp2)]
#    if (plotb == True):
#      fig, ax = plt.subplots(figsize=(10,7))
#      extent = 0,360,-90,90
#      im = ax.imshow(toPlot,'inferno',extent = extent,origin ='upper')
#      cb = fig.colorbar(im, fraction=0.0235, pad=0.04)
#      cb.set_label('H density [1/cc]',fontsize = 13)
#      ax.set_xlabel('Ecliptic Longitude [deg]')
#      ax.set_ylabel('Ecliptic Latitude [deg]')
          
#   return toPlot, H

#  if plane == 'meridional':
#    # verifying that arg should be between 0 to 360
#    if (arg<0) or (arg>360):
#      print('Azimuthal angle is outside the valid limits')
#      return -1
#    temp1 = abs(exosgrid.pvals - arg)
#    temp2 = np.argmin(temp1)
#    toPlot = H[:,int(temp2),:]
#    r     = np.linspace(exosgrid.rmin, exosgrid.rmax, int(exosgrid.numR))
#    theta = np.linspace(-np.pi/2, np.pi/2, int(exosgrid.numT))
#    R, Theta = np.meshgrid(r, theta) 
#    X1 = R*np.cos(Theta)
#    X2 = R*np.sin(Theta)
#    if (plotb==True):
#      fig, ax = plt.subplots(figsize=(4.5,9))
#      im = ax.pcolormesh(X1,X2,np.log10(toPlot),cmap='inferno',linewidth=0,rasterized = True)
#      cb = fig.colorbar(im, fraction=0.09, pad=0.04)
#      cb.set_label('log10(H density [1/cc])',fontsize = 13)
#      ax.axis('equal')
#      ax.axes.set_xlim(left=0, right=8) 
#      ax.axes.set_ylim(bottom=-8, top=8) 
#      ax.set_xlabel('X [RE]')
#      ax.set_ylabel('Z [RE]')
      
#    return toPlot, H

#  if plane == 'equatorial':
#    # arg is not needed, user can set it to 0
#    toPlot = H[int(exosgrid.numT/2),:,:]
#    r     = np.linspace(exosgrid.rmin, exosgrid.rmax, int(exosgrid.numR))
#    theta = np.linspace(0, 2*np.pi, int(exosgrid.numP)) 
#    R, Theta = np.meshgrid(r, theta) 
#    X1 = R*np.cos(Theta)
#    X2 = R*np.sin(Theta)
#    if (plotb == True):
#      fig, ax = plt.subplots(figsize=(9,9))
#      im = ax.pcolormesh(X1,X2,np.log10(toPlot),cmap='inferno',linewidth=0,rasterized = True)
#      cb = fig.colorbar(im, fraction=0.09, pad=0.04)
#      cb.set_label('log10(H density [1/cc])',fontsize = 13)
#      ax.axis('equal')
#      ax.axes.set_xlim(left=-8, right=8) 
#      ax.axes.set_ylim(bottom=-8, top=8) 
#      ax.set_xlabel('X [RE]')
#      ax.set_ylabel('Y [RE]')
      
#    return toPlot, H

def _add_half_shadow_disk(ax, center=(0.0, 0.0), radius=1.0, angle_deg=0.0,
                          edgecolor='k', lw=1.0, zorder=10):
    """
    Draw an Earth disk with half in shadow (black half-disk).
    angle_deg sets the shadow half orientation:
      0   -> shadow on +X side
      90  -> shadow on +Y side
      180 -> shadow on -X side
      -90 -> shadow on -Y side
    """
    x0, y0 = center

    # Outline of Earth
    circ = Circle((x0, y0), radius=radius, facecolor='none', edgecolor=edgecolor, lw=lw, zorder=zorder)
    ax.add_patch(circ)

    # Shadow half (a 180-degree wedge)
    # Wedge angles are in degrees, CCW from +x.
    # We'll fill the half centered on angle_deg (i.e., [angle-90, angle+90] is a half-plane),
    # but for a half-disk we use a 180-degree span.
    th1 = angle_deg - 90.0
    th2 = angle_deg + 90.0
    shadow = Wedge((x0, y0), r=radius, theta1=th1, theta2=th2, facecolor='k', edgecolor='none', zorder=zorder-1)
    ax.add_patch(shadow)


def _mollweide_forward(lon, lat):
    """
    Mollweide projection forward transform.
    Input:
      lon, lat in radians (lon in [-pi, pi], lat in [-pi/2, pi/2])
    Output:
      x, y in projected coordinates (units of radians-ish; consistent)
    """
    # Solve for theta: 2θ + sin(2θ) = π sin(lat)
    # Use Newton iterations; lat grid is modest so this is fine.
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # initial guess
    theta = lat.copy()

    rhs = np.pi * np.sin(lat)
    for _ in range(10):
        f = 2.0*theta + np.sin(2.0*theta) - rhs
        fp = 2.0 + 2.0*np.cos(2.0*theta)
        theta = theta - f / fp

    x = (2.0*np.sqrt(2.0)/np.pi) * lon * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    return x, y


def draw3DHmodel(model, exosgrid, plane, arg, plotb,
                 map_log10=False,
                 shadow_radius_re=1.0,
                 shadow_angle_equatorial_deg=0.0,
                 shadow_angle_meridional_deg=180.0):
    """
    Changes vs your original:
      - plane='map' now plots Mollweide projection.
      - equatorial & meridional optionally add half-shadow Earth disk.

    Parameters
    ----------
    map_log10 : bool
        If True, plot log10 in map view (like other planes).
    shadow_radius_re : float
        Earth disk radius in RE (usually 1.0).
    shadow_angle_equatorial_deg : float
        Orientation of shadow half in equatorial plane.
    shadow_angle_meridional_deg : float
        Orientation of shadow half in meridional plane.
    """

    H = generate3DHmodel(model, exosgrid)
    H = np.reshape(H, (int(exosgrid.numT), int(exosgrid.numP), int(exosgrid.numR)))

    # ---------- MAP (Mollweide) ----------
    if plane == 'map':
        if (arg < exosgrid.rmin) or (arg > exosgrid.rmax):
            print('Radius outside the valid limits')
            return -1

        temp2 = int(np.argmin(np.abs(exosgrid.rvals - arg)))
        toPlot = H[:, :, temp2]  # [theta_index, phi_index]

        if plotb:
            # Build lon/lat grids (assuming pvals: 0..360 deg, tvals: -90..90 deg)
            # Your imshow extent suggests lon in [0,360], lat in [-90,90].
            lon_deg = np.asarray(exosgrid.pvals, dtype=float)  # size numP
            lat_deg = np.asarray(exosgrid.tvals, dtype=float)  # size numT (latitude)

            # Convert to radians for Mollweide
            # Mollweide uses lon in [-pi, pi], so shift from [0,360] to [-180,180]
            lon_deg_wrapped = (lon_deg + 180.0) % 360.0 - 180.0
            lon = np.deg2rad(lon_deg_wrapped)
            lat = np.deg2rad(lat_deg)

            # Mesh (lat x lon) to match toPlot shape [numT, numP]
            Lon, Lat = np.meshgrid(lon, lat)
            X, Y = _mollweide_forward(Lon, Lat)

            Z = np.log10(toPlot) if map_log10 else toPlot

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.pcolormesh(X, Y, Z, cmap='inferno', shading='auto', rasterized=True)
            cb = fig.colorbar(im, fraction=0.03, pad=0.04)
            cb.set_label('log10(H density [1/cc])' if map_log10 else 'H density [1/cc]', fontsize=13)

            # Cosmetic: ticks in degrees
            # x range approx [-2.828, 2.828]; y range [-1.414, 1.414]
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

            # Add some tick labels that look like lon/lat
            # We'll place lon ticks at -150..150 every 60 deg
            lon_ticks_deg = np.array([-150, -90, -30, 30, 90, 150])
            lon_ticks = (2.0*np.sqrt(2.0)/np.pi) * np.deg2rad(lon_ticks_deg)  # approx x when theta=0
            ax.set_xticks(lon_ticks)
            ax.set_xticklabels([f"{d}°" for d in lon_ticks_deg])

            lat_ticks_deg = np.array([-60, -30, 0, 30, 60])
            lat_ticks = np.sqrt(2.0) * np.sin(np.deg2rad(lat_ticks_deg))  # approx y mapping
            ax.set_yticks(lat_ticks)
            ax.set_yticklabels([f"{d}°" for d in lat_ticks_deg])

            ax.set_title(f"Mollweide map at r={exosgrid.rvals[temp2]:.2f} RE")

        return toPlot, H

    # ---------- MERIDIONAL ----------
    if plane == 'meridional':
        if (arg < 0) or (arg > 360):
            print('Azimuthal angle is outside the valid limits')
            return -1

        temp2 = int(np.argmin(np.abs(exosgrid.pvals - arg)))
        toPlot = H[:, temp2, :]  # [theta, r]

        r = np.linspace(exosgrid.rmin, exosgrid.rmax, int(exosgrid.numR))
        theta = np.linspace(-np.pi/2, np.pi/2, int(exosgrid.numT))
        R, Theta = np.meshgrid(r, theta)
        X1 = R*np.cos(Theta)
        X2 = R*np.sin(Theta)

        if plotb:
            fig, ax = plt.subplots(figsize=(4.5, 9))
            im = ax.pcolormesh(X1, X2, np.log10(toPlot), cmap='inferno', linewidth=0, rasterized=True, shading='auto')
            cb = fig.colorbar(im, fraction=0.09, pad=0.04)
            cb.set_label('log10(H density [1/cc])', fontsize=13)

            ax.axis('equal')
            ax.set_xlim(0, 8)
            ax.set_ylim(-8, 8)
            ax.set_xlabel('X [RE]')
            ax.set_ylabel('Z [RE]')

            # Earth + half-shadow disk at origin
            _add_half_shadow_disk(ax, center=(0.0, 0.0), radius=shadow_radius_re,
                                  angle_deg=shadow_angle_meridional_deg)

        return toPlot, H

    # ---------- EQUATORIAL ----------
    if plane == 'equatorial':
        toPlot = H[int(exosgrid.numT/2), :, :]  # [phi, r]

        r = np.linspace(exosgrid.rmin, exosgrid.rmax, int(exosgrid.numR))
        theta = np.linspace(0, 2*np.pi, int(exosgrid.numP))
        R, Theta = np.meshgrid(r, theta)
        X1 = R*np.cos(Theta)
        X2 = R*np.sin(Theta)

        if plotb:
            fig, ax = plt.subplots(figsize=(9, 9))
            im = ax.pcolormesh(X1, X2, np.log10(toPlot), cmap='inferno', linewidth=0, rasterized=True, shading='auto')
            cb = fig.colorbar(im, fraction=0.09, pad=0.04)
            cb.set_label('log10(H density [1/cc])', fontsize=13)

            ax.axis('equal')
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_xlabel('X [RE]')
            ax.set_ylabel('Y [RE]')

            # Earth + half-shadow disk at origin
            _add_half_shadow_disk(ax, center=(0.0, 0.0), radius=shadow_radius_re,
                                  angle_deg=shadow_angle_equatorial_deg)

        return toPlot, H

    print("plane must be 'map', 'meridional', or 'equatorial'")
    return -1

#-------------------------------------------------------------------------------
def draw1DHmodel(model, minrad = 3, maxrad=10, radstep = 0.1,plotb = False):
  if (model=='C19M03'):
    # From XMM-Newton
    # Verify boundaries
    if (minrad<3) or (maxrad>10):
      print('Radial limits out of the limits')
      return -1
    Radius = np.arange(minrad,maxrad+radstep,radstep,dtype=np.float64)
    No = 39.9
    N = (No*10**3)/(Radius**3)
    if (plotb == True):
      fig, ax = plt.subplots(figsize=(10,7))
      ax.plot(Radius,N,linewidth=2)
      ax.set_xlabel('Geocentric Distance [RE]')
      ax.set_ylabel('H density [1/cc]')
      ax.grid('on')
      ax.axis('tight')
      ax.axes.set_xlim(left=minrad, right=maxrad)     
    return N,Radius
  
  if (model == 'C19O01'):
    # From XMM-Newton
    # Verify boundaries
    if (minrad<3) or (maxrad>10):
      print('Radial limits out of the limits')
      return -1
    Radius = np.arange(minrad,maxrad+radstep,radstep,dtype=np.float64)
    No = 57.6
    N = (No*10**3)/(Radius**3)
    if (plotb == True):
      fig, ax = plt.subplots(figsize=(10,7))
      ax.plot(Radius,N,linewidth=2)
      ax.set_xlabel('Geocentric Distance [RE]')
      ax.set_ylabel('H density [1/cc]')
      ax.grid('on')
      ax.axis('tight')
      ax.axes.set_xlim(left=minrad, right=maxrad)     
    return N,Radius

  if (model == 'J22'):   
    # From XMM-Newton
    # Verify boundaries
    if (minrad<3) or (maxrad>10):
      print('Radial limits out of the limits')
      return -1
    Radius = np.arange(minrad,maxrad+radstep,radstep,dtype=np.float64)
    No = 36.8
    N = (No*10**3)/(Radius**3)
    if (plotb == True):
      fig, ax = plt.subplots(figsize=(10,7))
      ax.plot(Radius,N,linewidth=2)
      ax.set_xlabel('Geocentric Distance [RE]')
      ax.set_ylabel('H density [1/cc]')
      ax.grid('on')
      ax.axis('tight')
      ax.axes.set_xlim(left=minrad, right=maxrad)     
    return N,Radius 


#-------------------------------------------------------------------------------
def ReadingSourceFile(buf):
  f     = open(buf,'r')
  line = f.readline() # Irradiance header
  line = f.readline() # Irradiance value
  Irradiance = float(line[0:8])
  line = f.readline() # Title header
  DataSF = [] 

  while True:  
    # Get next line from file
    line = f.readline()

    # if line is empty
    # end of file is reached
    if not line:
        break
    #print("Line{}: {}".format(count, line.strip()))
    Altitude_t  = float(line[0:13])
    SZA_t       = float(line[14:22])
    Temp_t      = float(line[23:40])
    O2_t        = float(line[41:70])
    H_t         = float(line[71:100])
    S_single_t  = float(line[101:125])
    S_mult_t    = float(line[126:150])
    DataSF.append([Altitude_t,SZA_t,Temp_t,O2_t,H_t,S_single_t,S_mult_t])

  f.close()

  # Converting list into numpy array
  DataSF_np = np.array(DataSF)

  # Getting data in columns
  Altitude  = DataSF_np[:,0]
  SZA       = DataSF_np[:,1]
  Temp      = DataSF_np[:,2]
  O2_dens   = DataSF_np[:,3] 
  H_dens    = DataSF_np[:,4] 
  S_single  = DataSF_np[:,5]
  S_mult    = DataSF_np[:,6]

  Altitude  = np.unique(Altitude)
  SZA       = np.unique(SZA)
  AltLen    = len(Altitude)
  SZALen    = len(SZA)
  O2_dens   = O2_dens[0:len(Altitude)]
  H_dens    = H_dens[0:len(Altitude)]
  Temp      = Temp[0:len(Altitude)]

  # Reshape of S_single for plots
  SS = np.reshape(S_single,(SZALen,AltLen))
  SM = np.reshape(S_mult,(SZALen,AltLen))

  return Altitude,SZA,Temp,O2_dens,H_dens,SS,SM,Irradiance

#-------------------------------------------------------------------------------
def CalculateLOSfromSourceFunction(sat_pos,sat_los,Altitude,SZA,Temp,O2_dens,H_dens,SS,SM,Irradiance,dl = 1):
  Xdir    = np.array([1,0,0])
  pf_ang  = np.arccos(Xdir.dot(sat_los)/(np.linalg.norm(Xdir)*np.linalg.norm(sat_los)))
  pf      = (11./12.) + ((1./4.) * 0.5 * (np.cos(2*pf_ang) +1))
  [azim,elev,radius] = cart2sph(sat_pos[0],sat_pos[1],sat_pos[2])
  
  lyman_alpha = 121.6e-9 # m
  lightspeed  = 3e8 # m/s
  planck      = 6.63e-34 # J.s
  f_flux      = (Irradiance*lyman_alpha)*(1e-4)/(planck*lightspeed) #ph/s/m2
  g_factor    = 3.47e-4*(f_flux/1e11)**(1.21) # 1/s

  # Initial Values
  num_densO2_old  = 0
  num_densH_old   = 0
  tot_tauh        = 0
  tot_tauO2       = 0
  sigma_0         = (5.96e-12)/np.sqrt(1000.0) #>>>> CHANGE IT WITH DATA FROM FILE
  intensity       = 0
  current_pos     = sat_pos
  current_pos     = current_pos*6371*1e5 # in cm
  maxRAD          = max(Altitude)

  fH  = interpolate.interp1d(Altitude*1e5, H_dens)
  fO2 = interpolate.interp1d(Altitude*1e5,O2_dens)
  fSS = interpolate.interp2d(Altitude*1e5,SZA,SS)
  fSM = interpolate.interp2d(Altitude*1e5,SZA,SM)
  fT  = interpolate.interp1d(Altitude*1e5,Temp)

  Temp_O2LY   = np.array([84.0,203.0,288.0,366.0,1500.0])
  CS_O2LY     = np.array([8.96838e-21, 8.71880e-21, 9.48889e-21, 1.13590e-20,1.13590e-20])
  fSigmaO2LY  = interpolate.interp1d(Temp_O2LY,CS_O2LY)

  ds_max = 500e5 # in cm

  # Main Loop for a LOS
  while True:
    radius      = np.sqrt(current_pos[0]**2 + current_pos[1]**2 + current_pos[2]**2)  # in geocentric cm
    rad_alt     = radius - 6371*1e5 # altitude

    if rad_alt > maxRAD*1e5 :
      break

    current_pos_uv  = current_pos/np.linalg.norm(current_pos)
    sza             = np.arccos(Xdir.dot(current_pos_uv)/(np.linalg.norm(Xdir)*np.linalg.norm(current_pos_uv)))*180/np.pi

    ## Getting H and O2 densities through interpolation
    num_densH   = fH(rad_alt)
    num_densO2  = fO2(rad_alt)
    #print(num_densH)
    ## Getting Temperature & sigma_0
    temp        = fT(rad_alt)
    sigma_0     = (5.96e-12)/np.sqrt(temp)
    sigma_O2    = fSigmaO2LY(temp)

    ## Getting tauH and tauO2
    tot_tauh    = tot_tauh + (num_densH + num_densH_old)*0.5*sigma_0*dl
    tot_tauO2   = tot_tauO2 + (num_densO2 + num_densO2_old)*0.5*sigma_O2*dl

    ## Getting HOLSTEIN calculation
    hx = getHolstein(tot_tauh)

    ## Getting Source functions
    s0x = fSS(rad_alt,sza)*g_factor
    snx = fSM(rad_alt,sza)*g_factor

    ## Getting Intensity
    intensity   = intensity + pf*s0x*hx*np.exp(-tot_tauO2)*dl + pf*(snx)*hx*np.exp(-tot_tauO2)*dl

    ## Updating dl
    if (num_densH == 0) or (sigma_0 == 0):
      ds = ds_max
    else:
      ds = 0.5/(num_densH*sigma_0)

    if (ds>= ds_max):
      dl = ds_max
    else:
      dl = ds

    ## Updating num dens
    num_densH_old   = num_densH
    num_densO2_old  = num_densO2

    # Update the current position
    current_pos = current_pos + dl*sat_los #in RE
  
  return intensity
