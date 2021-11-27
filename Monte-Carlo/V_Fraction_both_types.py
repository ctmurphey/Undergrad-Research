import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from scipy import integrate
import math

R_sun = 8.5# kpc, distance from galactic center to sun
R_thin = 2.9 # kpc, scale radius of Milky Way
h_thin = 0.095 # kpc, scale height of Milky Way
minvis = np.arange(60, 0, -0.05) #minimum visible magnitude


# STEP 1: generate random supernovae
sn_num = 1000 # number of random supernovae generated
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



#r = R_thin*np.log(1/(1-mu1)) # exponential radial distribution
r = R_thin*o
theta = 2*np.pi*mu2          # uniform angular distribution from 0 to 2pi
z = h_thin*np.log(1/1-mu3)   # exponential height distribution

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

    rho = np.exp(-r/R_thin)*np.exp(-abs(z)/h_thin)
    rho = rho / (R_thin * (1. - np.exp(-R_sun/R_thin)))
    return rho


def dAv_dr(radius, l_rad, b_rad):
    "Extinction Rate due to Dust"
    z_cyl = radius * np.sin(b_rad) # solarcentric radius component in plane
    r_par = radius * np.cos(b_rad) # Galactocentric cylindrical radius
    R_cyl = np.sqrt(r_par**2 - 2.*R_sun*r_par*np.cos(l_rad) + R_sun**2)

    Av_gc = 30.0
    dAv_dr = Av_gc * rho_dust(R_cyl,z_cyl)
    #Av_gc = 0
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

def make_line(magnitude, lab_string):
    "Plots line of probabilty vs minimum visible magnitude, keeping the absolute magnitude constant"

    
    extinct = np.zeros_like(minvis)

    for i in range(len(minvis)):
        extinct[i] = magnitude - minvis[i] #total exinction along sightline needed
        dimmed = np.zeros_like(radius)
        for i in range(len(radius)):
            dimmed[i] = distmod(radius[i]) + dustdim(l[i], b[i], radius[i])
    
    # STEP 4: sort visible and invisible supernova
    fraction = np.zeros_like(extinct)
    lowerr = np.zeros_like(extinct)
    higherr = np.zeros_like(extinct)
    for i in range(len(extinct)):
        mask = dimmed > -extinct[i]

        newmask = ~mask
        brightx = x[newmask]
        brighty = y[newmask] # visible supernovae
        brightz = z[newmask]

        brightl = l_deg[newmask]
        brightb = b_deg[newmask]


        fraction[i] = len(brightx)/len(radius)*100
        lowerr[i] = len(brightx)/len(radius)*100*(1 - np.sqrt(1/len(brightx)+1/len(radius)))
        higherr[i] = len(brightx)/len(radius)*100*(1 + np.sqrt(1/len(brightx)+1/len(radius)))

    plt.plot(minvis, fraction, label = lab_string)
    plt.plot(minvis, lowerr)
    plt.plot(minvis, higherr)

make_line(-18, "Type Ia")
make_line(-16, "Core Collapse")

string = str(sn_num) + ' Supernovae Simulated'
plt.plot((26, 26), (0,100))
plt.plot((6, 6), (0, 100))
plt.title('Percentage of Supernovae Seen vs Magnitude', weight = 'bold')
plt.xlabel('Minimum Visible Magnitude', weight = 'bold')
plt.ylabel('Percentage of Supernovae Seen', weight = 'bold')
plt.xlim(np.max(minvis), np.min(minvis))
plt.ylim(0, 100)
plt.annotate(string, (20, 10), weight = 'bold')
plt.annotate("LSST Limiting G-Band Magnitude", (25.8, 25), weight = 'bold', size = 8)
plt.annotate("Limiting Naked-Eye Magnitude", (15, 81), weight = 'bold')
plt.grid()
plt.legend()
plt.show()