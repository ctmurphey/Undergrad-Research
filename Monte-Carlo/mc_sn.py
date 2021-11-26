import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from scipy import integrate
import math
#### NEED TO FIX ADAMS DISTRIBUTION###
# 1: get random sets of points in galaxy -> copy procedure from randgalaxy file
# 2: find relative location to sun (need distance, l, and b)
# 3: make new 1d array of total dimming (integral + distance modulus)
# 4: use new array to mask values greater than, then less than, M-m
### be sure to apply mask to all x, y, and z values
# 5: plot in 3d, different colors for those we can and cannot see
# 6: plot in 2d in sky, bin via latitude and longitude
# after it works, add points until it takes annoyingly long to run (more the merrier!)
 
R_sun = 8.5# kpc, distance from galactic center to sun
R_thin = 2.9 # kpc, scale radius of Milky Way
h_thin = 0.095 # kpc, scale height of Milky Way

absmag = -18 # Absolute magnitude of a typical supernova
minvis = 5 # dimmest apparent magnitude (26 for LSST, 5 or 6 for visible)
minvis = 26 # magnitude of venus -- make it brightest thing in sky
extinct = absmag - minvis #total exinction along sightline needed




# STEP 1: generate random supernovae
sn_num = 10000 # number of random supernovae generated
mu1 = rand(sn_num) # random values to use for r, theta, and z
mu2 = rand(sn_num)
mu3 = rand(sn_num)

SN_dist = 'Adams'

def radius_dist(y,u):
    return np.log((y+1)/(1-u))

a = np.zeros(sn_num)
k = [0.5]*sn_num
o = np.zeros(sn_num)
for i in range(0,sn_num):
    while a[i] != 1:
        o[i] =  radius_dist(k[i],mu1[i])
        if abs(o[i]-k[i]) <= 0.0001:
            a[i] = 1
        else:
            k[i] = o[i]


###Need to fix here:
r = r = R_thin*o             # exponential radial distribution
theta = 2*np.pi*mu2          # uniform angular distribution from 0 to 2pi
z = h_thin*np.log(1/(1-mu3)) # exponential height distribution



x = np.zeros_like(r)
y = np.zeros_like(r)

for i in range(len(mu1)):
    x[i] = r[i]*np.cos(theta[i]) # used for plotting in step 5
    y[i] = r[i]*np.sin(theta[i]) # same use as x array




# STEP 2: find relative position to Sun
    ### set sun at x = r_sun, y = z = 0

l = np.zeros_like(x) # galactic longitude w.r.t. Sun
b = np.zeros_like(y) # galactic latitude w.r.t. Sun

radius = np.sqrt((x-R_sun)**2 + y**2 + z**2)
for i in range(len(radius)):
    if x[i] <= R_sun: #arcsin only covers from -pi/2 to pi/2
        l[i] = np.arcsin(y[i] / radius[i])
        
    else: # cover the rest of the sky (whole thing goes from -pi/2 to 3pi/2)
        l[i] = np.arcsin(y[i] / radius[i]) + np.pi
    b[i] = np.arcsin(z[i] / radius[i])
   

#getting arrays in degrees and centering around 0 for plotting
l_deg = np.zeros_like(l)
b_deg = np.zeros_like(b)
for i in range(len(l)):
    l_deg[i] = math.degrees(l[i])
    b_deg[i] = math.degrees(b[i])
for j in range(len(l_deg)):
    if l_deg[j] > 180:
        l_deg[j] = l_deg[j]-360



# STEP 3: get dimming for each supernova
def rho_dust(r, z):
    "Dust Density Function"
    R_thin = 2.9 # kpc
    h_thin = 0.095 # kpc

    rho = np.e**(-r/R_thin)*np.e**(-abs(z)/h_thin)
    rho = rho / (R_thin * (1. - np.e**(-R_sun/R_thin)))
    return rho


def dAv_dr(radius, l_rad, b_rad):
    "Extinction Rate due to Dust"
    z_cyl = radius * np.sin(b_rad) # solarcentric radius component in plane
    r_par = radius * np.cos(b_rad) # Galactocentric cylindrical radius
    R_cyl = np.sqrt(r_par**2 - 2.*R_sun*r_par*np.cos(l_rad) + R_sun**2)

    Av_gc = 30.0
    dAv_dr = Av_gc * rho_dust(R_cyl,z_cyl)
    
    return dAv_dr

def distmod(rad):
    "Distance Modulus, takes radii in kpc"
    return 5*np.log10(rad*1000/10)

def dustdim(l_rad, b_rad, radi):
    "Total dimming due to dust"
    Sigfunct = lambda r: dAv_dr(r, l_rad, b_rad) #so dAv_dr will intgrate correctly
    Sigma, err = integrate.quad(Sigfunct,0 ,radi) #get magnitude loss due to extinction
    return Sigma

dimmed = np.zeros_like(radius)
for i in range(len(radius)):
    dimmed[i] = distmod(radius[i]) + dustdim(l[i], b[i], radius[i])


# STEP 4: sort visible and invisible supernova
mask = dimmed > -extinct
darkx = x[mask] 
darky = y[mask] # invisible supernovae
darkz = z[mask]

newmask = ~mask
brightx = x[newmask]
brighty = y[newmask] # visible supernovae
brightz = z[newmask]

brightl = l_deg[newmask]
brightb = b_deg[newmask]


# STEP 5: plot each in 3D, print out visible fraction and whatever other ouputs seem appropriate
fraction = len(brightx)/len(radius)
print("The fraction of seen supernovae is", fraction)

title = "Simulation of " + str(sn_num) + " Random Supernovae in the Milky Way"
fig = plt.figure()
fig.suptitle(title, fontsize = 16)

ax = fig.add_subplot(221, projection = '3d') #3d map of galaxy
ax.plot(darkx, darky, darkz, 'k,', label= "Missed (black)")
ax.plot(darkx, darky, -darkz, 'k,')
ax.plot(brightx, brighty, brightz, 'b,', label = "Seen (blue)")
ax.plot(brightx, brighty, -brightz, 'b,')
ax.plot([R_sun], [0], [0], 'yo', label = "Sun") # Using the Sun as a reference point
ax.set_title("Milky Way Map of Both Seen and Invisible Supernovae", weight = 'bold')
ax.legend(loc = 4)

ax2 = fig.add_subplot(222) # sky plot
text = str(round(fraction*100, 4)) + " percent of Supernovae are visible"
ax2.plot(brightl, brightb, 'r,')
ax2.plot(brightl, -brightb, 'r,')
ax2.set_title("Location of Randomly-Generated Seen Supernovae on the Sky", weight = 'bold')
ax2.set_xlabel("Galactic Longitude (l)", weight = 'bold')
ax2.set_ylabel("Galactic Latitude (b)", weight = 'bold')
ax2.text(-80, 40, text)

ax3 = fig.add_subplot(223) # Longitude histogram
ax3.hist(brightl, density = True, bins = 360)
ax3.set_title("Longitude Dependence of Seen Supernovae", weight = 'bold')
ax3.set_xlabel(" Galactic Longitude (l)", weight = 'bold')
ax3.set_ylabel("Probability per Degree", weight = 'bold')
ax3.set_xlim(-180, 180)

ax4 = fig.add_subplot(224) # Latitude histogram
ax4.hist([brightb, -brightb], density = True, bins = 90)
ax4.set_title("Latitude Dependence of Seen Supernovae", weight = 'bold')
ax4.set_xlabel("Galactic Latitude (b)", weight = 'bold')
ax4.set_ylabel("Probability per Degree")
ax4.set_xlim(-45, 45)

plt.show()