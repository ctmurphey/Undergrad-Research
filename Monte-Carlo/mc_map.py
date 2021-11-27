import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from scipy import integrate
import math
from scipy.stats import kde
from sys import exit
from time import time

t0 = time()
# Variables, numbers and strings
R_sun = 8.5# kpc, distance from galactic center to sun
h_sun = 0.02 #kpc, height of sun above galactic plane
R_thin = 2.9 # kpc, scale radius of Milky Way
R_thin = 5 # For tweaking, didn't want to remove orginal value
h_thin = 0.095 # kpc, scale height of Milky Way
R_thick = 2.4 #kpc
h_thick = 0.8#kpc
sn_num = int(2E4) # number of random supernovae generated
band = "V" #band we're looking in
SN_type = "CC" #sn type we're looking at
minvis = 2 # 2: easily noticeable everywhere; 6: noticeable in dark spaces; 7: human vision limit; 26: LSST


if band == "V":
    A_gc = 30
    if SN_type == "CC": absmag = -16
    elif SN_type == "Ia": absmag = -18.5

elif band == "K":
    A_gc = 3.51
    if SN_type == "CC": absmag = -17
    elif SN_type == "Ia": absmag = -17.8

else:
    exit("Invalid band, please use either V or K band")



extinct = absmag - minvis #total exinction along sightline needed




# STEP 1: generate random supernovae
mu1 = rand(sn_num) # random values to use for r, theta, and z
mu2 = rand(sn_num)
mu3 = rand(sn_num)


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


r = np.zeros_like(o)
theta = np.zeros_like(o)
z = np.zeros_like(o)
if SN_type == 'CC': # Thin disk
    for i in range(len(o)):
        if i%2 == 0:
            r[i] = R_thin*o[i]
            theta[i] = 2*np.pi*mu2 [i]         # uniform angular distribution from 0 to 2pi
            z[i] = h_thin*np.log(1/1-mu3[i])   # exponential height distribution
        elif i%2 == 1:
            r[i] = R_thin*o[i]
            theta[i] = 2*np.pi*mu2 [i]         # uniform angular distribution from 0 to 2pi
            z[i] = -h_thin*np.log(1/1-mu3[i])   # exponential height distribution
elif SN_type == 'Ia': #Thin + Thick disks
    for i in range(len(o)):
        if i%4 == 0:
            r[i] = R_thin*o[i]
            theta[i] = 2*np.pi*mu2 [i]         # uniform angular distribution from 0 to 2pi
            z[i] = h_thin*np.log(1/1-mu3[i])   # exponential height distribution
        elif i%4 == 1:
            r[i] = R_thick*o[i]
            theta[i] = 2*np.pi*mu2 [i]         # uniform angular distribution from 0 to 2pi
            z[i] = h_thick*np.log(1/1-mu3[i])   # exponential height distribution
        elif i%4 == 2:
            r[i] = R_thin*o[i]
            theta[i] = 2*np.pi*mu2 [i]         # uniform angular distribution from 0 to 2pi
            z[i] = -h_thin*np.log(1/1-mu3[i])   # exponential height distribution
        elif i%4 == 3:
            r[i] = R_thick*o[i]
            theta[i] = 2*np.pi*mu2 [i]         # uniform angular distribution from 0 to 2pi
            z[i] = -h_thick*np.log(1/1-mu3[i])   # exponential height distribution
else:
    exit("Invalid SN type, make sure it is either CC or Ia")

print(sn_num, "supernovae generated...")




x = np.zeros_like(r)
y = np.zeros_like(r)

for i in range(len(mu1)):
    x[i] = r[i]*np.cos(theta[i]) # used for plotting in step 5
    y[i] = r[i]*np.sin(theta[i]) # same use as x array






# STEP 2: find relative position to Sun
    ### set sun at x = r_sun, y = z = 0

l = np.zeros_like(x) # galactic longitude w.r.t. Sun
b = np.zeros_like(y) # galactic latitude w.r.t. Sun

radius = np.sqrt((x-R_sun)**2 + y**2 + (z-h_sun)**2)
for i in range(len(radius)):
    if x[i] <= R_sun: #arcsin only covers from -pi/2 to pi/2
        l[i] = np.arcsin(y[i] / np.sqrt((x[i]-R_sun)**2 + y[i]**2))
        
    else: # cover the rest of the sky (whole thing goes from -pi/2 to 3pi/2)
        l[i] = np.arcsin(y[i] / np.sqrt((x[i]-R_sun)**2 + y[i]**2)) + np.pi
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

print('Extinction calculated...')





# STEP 4: sort visible and invisible supernova
mask = dimmed > -extinct

newmask = ~mask

brightl = l_deg[newmask] #visible supernovae
brightb = b_deg[newmask]
print('Supernovae sorted...')
print('%1.2f percent of supernovae have an apparent magnitude greater than %d' % (len(brightl)/sn_num*100, minvis))

x_pts = brightl[np.isfinite(brightl) & np.isfinite(brightb)] #gets rid of NANs
y_pts = brightb[np.isfinite(brightl) & np.isfinite(brightb)]



k = kde.gaussian_kde([x_pts, y_pts])


xi, yi = np.mgrid[-180:180:360*6*1j, -10:10:20*60*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))*360
 
# Make the plot
print('Plotting supernovae...')
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.jet)
plt.title('%d %s Supernovae in the %s-Band' %(sn_num, SN_type, band))
plt.colorbar(label = 'Percent chance per Square Degree')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.scatter([4.5, 120.1, 175.4, 130.7], [6.8, 1.4, -5.8, 3.1], c='w')
#plt.annotate('SN1006', (-32.4, 14.6), color = 'w')
plt.annotate('SN1604 (Kepler)',(4.5, 6.8), color = 'w')
plt.annotate('SN1572 (Tycho)', (120.1, 1.4), color = 'w')
plt.annotate('SN1054 (Crab Nebula)', (-175.4, -5.8), color = 'w')
plt.annotate('SN1181 (3C-58)',(130.7,3.1), color = 'w')

t1 = time()
hours = (t1-t0)/3600
minutes = (hours - int(hours))*60
seconds = (minutes-int(minutes))*60

print('Time to run: %d hours %d minutes %.1f seconds' %(int(hours), int(minutes), seconds))
plt.show()
