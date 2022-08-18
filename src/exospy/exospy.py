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
def get_density(model,radius,theta,phi):
  # Verifying they are column vectors
  #if theta.shape[1] > theta.shape[0]:
  #  theta = np.transpose(theta)

  #if phi.shape[1] > phi.shape[0]:
  #  phi = np.transpose(phi)

  #if radius.shape[1] > radius.shape[0]:
  #  radius = np.transpose(radius)

  n_h = 0 
  n_radii = 1#radius.shape[0]

  N     = np.zeros([n_radii,1])
  A_lm  = np.zeros([10,n_radii])
  B_lm  = A_lm
  Y_lm  = 0 #np.zeros([theta.shape[0],1])

  N     = N_coefficients(model,radius)
  AB    = AB_coefficients(model,radius); 
  Y_lm  = spherical_harmonics(theta);

  A_lm  = AB[0:10,:]
  B_lm  = AB[10:20,:] 

  #l = 0, m = 0
  n_h = n_h + (A_lm[0,:]*np.cos(0*phi)+B_lm[0,:]*np.sin(0*phi))*Y_lm[0,:]
  #l = 1, m = 0, 1
  n_h = n_h + (A_lm[1,:]*np.cos(0*phi)+B_lm[1,:]*np.sin(0*phi))*Y_lm[1,:] + (A_lm[2,:]*np.cos(1*phi)+B_lm[2,:]*np.sin(1*phi))*Y_lm[2,:]
  #l = 2, m = 0, 1
  n_h = n_h + (A_lm[3,:]*np.cos(0*phi)+B_lm[3,:]*np.sin(0*phi))*Y_lm[3,:] + (A_lm[4,:]*np.cos(1*phi)+B_lm[4,:]*np.sin(1*phi))*Y_lm[4,:]
  #l = 2, m = 2
  n_h = n_h + (A_lm[5,:]*np.cos(2*phi)+B_lm[5,:]*np.sin(2*phi))*Y_lm[5,:]
  #l = 3, m = 0, 1
  n_h = n_h + (A_lm[6,:]*np.cos(0*phi)+B_lm[6,:]*np.sin(0*phi))*Y_lm[6,:] + (A_lm[7,:]*np.cos(1*phi)+B_lm[7,:]*np.sin(1*phi))*Y_lm[7,:]
  #l = 3, m = 2, 3
  n_h = n_h + (A_lm[8,:]*np.cos(2*phi)+B_lm[8,:]*np.sin(2*phi))*Y_lm[8,:] + (A_lm[9,:]*np.cos(3*phi)+B_lm[9,:]*np.sin(3*phi))*Y_lm[9,:]

  density = n_h*N*np.sqrt(4*np.pi);

  return density

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
def generate3DHmodel(model,exosgrid):
  H = np.zeros((int(exosgrid.numvoxels),1))
  theta = np.array([[0.0]])
  rad = np.array([[0.0]])
  phi = np.array([[0.0]])

  for t_id in range(int(exosgrid.numT)):
    theta[0,0] = exosgrid.tvals[t_id]+exosgrid.tstep/2
    theta[0,0] = theta[0,0]*np.pi/180
    for p_id in range(int(exosgrid.numP)):
      phi[0,0] = exosgrid.pvals[p_id]+exosgrid.pstep/2
      phi[0,0] = phi[0,0]*np.pi/180
      for r_id in range(int(exosgrid.numR)):
        rad[0,0] = exosgrid.rvals[r_id]+exosgrid.rstep/2
        H[int(r_id+p_id*exosgrid.numR+t_id*exosgrid.numR*exosgrid.numP),0] = get_density(model,rad,theta,phi)
        #print(rad,theta,phi)

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
      intensity = intensity + (pf*g_factor/10.0**6)*get_density(model,radius,colat*np.pi/180,phi*np.pi/180)*dl*(6371*10**5)
      # Update the current position
      current_pos = current_pos + dl*r_los[i,:] #in RE

    # Save Intensity
    Intensity_v[i,0] = intensity
  
  return Intensity_v

#-------------------------------------------------------------------------------
def draw3DHmodel(model,exosgrid,plane,arg,plotb):
  H = generate3DHmodel(model,exosgrid)
  H = np.reshape(H,(int(exosgrid.numT), int(exosgrid.numP), int(exosgrid.numR)))

  if plane == 'map':
    # verifying that arg should be between exosgrid.rmin and exosgrid.rmax
    if (arg<exosgrid.rmin) or (arg>exosgrid.rmax):
      print('Radius outside the valid limits')
      return -1
    temp1 = abs(exosgrid.rvals - arg)
    temp2 = np.argmin(temp1)
    toPlot = H[:,:,int(temp2)]
    if (plotb == True):
      fig, ax = plt.subplots(figsize=(10,7))
      extent = 0,360,-90,90
      im = ax.imshow(toPlot,'inferno',extent = extent,origin ='upper')
      cb = fig.colorbar(im, fraction=0.0235, pad=0.04)
      cb.set_label('H density [1/cc]',fontsize = 13)
      ax.set_xlabel('Ecliptic Longitude [deg]')
      ax.set_ylabel('Ecliptic Latitude [deg]')
          
    return toPlot, H

  if plane == 'meridional':
    # verifying that arg should be between 0 to 360
    if (arg<0) or (arg>360):
      print('Azimuthal angle is outside the valid limits')
      return -1
    temp1 = abs(exosgrid.pvals - arg)
    temp2 = np.argmin(temp1)
    toPlot = H[:,int(temp2),:]
    r     = np.linspace(exosgrid.rmin, exosgrid.rmax, int(exosgrid.numR))
    theta = np.linspace(-np.pi/2, np.pi/2, int(exosgrid.numT))
    R, Theta = np.meshgrid(r, theta) 
    X1 = R*np.cos(Theta)
    X2 = R*np.sin(Theta)
    if (plotb==True):
      fig, ax = plt.subplots(figsize=(4.5,9))
      im = ax.pcolormesh(X1,X2,np.log10(toPlot),cmap='inferno',linewidth=0,rasterized = True)
      cb = fig.colorbar(im, fraction=0.09, pad=0.04)
      cb.set_label('log10(H density [1/cc])',fontsize = 13)
      ax.axis('equal')
      ax.axes.set_xlim(left=0, right=8) 
      ax.axes.set_ylim(bottom=-8, top=8) 
      ax.set_xlabel('X [RE]')
      ax.set_ylabel('Z [RE]')
      
    return toPlot, H

  if plane == 'equatorial':
    # arg is not needed, user can set it to 0
    toPlot = H[int(exosgrid.numT/2),:,:]
    r     = np.linspace(exosgrid.rmin, exosgrid.rmax, int(exosgrid.numR))
    theta = np.linspace(0, 2*np.pi, int(exosgrid.numP)) 
    R, Theta = np.meshgrid(r, theta) 
    X1 = R*np.cos(Theta)
    X2 = R*np.sin(Theta)
    if (plotb == True):
      fig, ax = plt.subplots(figsize=(9,9))
      im = ax.pcolormesh(X1,X2,np.log10(toPlot),cmap='inferno',linewidth=0,rasterized = True)
      cb = fig.colorbar(im, fraction=0.09, pad=0.04)
      cb.set_label('log10(H density [1/cc])',fontsize = 13)
      ax.axis('equal')
      ax.axes.set_xlim(left=-8, right=8) 
      ax.axes.set_ylim(bottom=-8, top=8) 
      ax.set_xlabel('X [RE]')
      ax.set_ylabel('Y [RE]')
      
    return toPlot, H

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