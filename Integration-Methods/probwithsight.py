#step 1: get distances using code from sightextinct
#step 2: use these radii for function in sky distribution code
#step 3: sum over all sightlines to get probability of supernova being visible

#Note: probability will only be for one quadrant: multiply by 4 to get actual probability

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from math import gamma
import math

#Step 1: get distances for sightlines
#absmag = -18 #dimmest absolute magnitude of a supernova
absmag = -18
minvis = 6 # dimmest apparent magnitude (26 for LSST, 6 for visible)
extinct = absmag - minvis #total exinction along sightline needed

R_sun = 8.5 #kpc

l_max = 180 #max longitude
b_max = 90 #max latitude
n_b = 30 #number of latitude points on plot
n_l = 30 #number of longitude points

## model parameters:
## supernova type
SN_type = 'CC'
#SN_type = 'Ia'

## supernova density distribution model
SN_dist = 'spherical'
SN_dist = 'thindisk'
SN_dist = 'Adams'
SN_dist = 'Green'

## plot region is zoomed in longitidue
zoom = True
zoom = False

# geometry parameters
R_sun = 8.5# kpc
R_cc = 2.9# kpc
R_ia = 2.4# kpc
h_cc = 0.05# kpc
h_ia = 0.8 # kpc

if (zoom):
    l_max_deg = 90.
else:
    l_max_deg = 180.



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
    Av_gc = 0
    dAv_dr = Av_gc * rho_dust(R_cyl,z_cyl)
    
    return dAv_dr

def distfinder(l,b): 
    "Finds Distance Based on Magnitudes"
    "Accepts degrees for l and b"
    b_rad = math.radians(b)
    l_rad = math.radians(l)
    step_length = 0.01 # kpc

    startpoint = 0. #where to start integral (kpc)
    endpoint   = 0. #where to end integral (kpc)
    
    newapp = absmag #creating apparent magnitude
    maxvis = 0. #maximum visible distace (pc)
    blocked = 0. # sum of total light lost to extinction, to keep from having to integrate over and over

    while newapp < minvis: #while object is still visible in sky...
        if endpoint > 40: #setting max distance for praticality purposes
                break
        startpoint = endpoint #shift where code will integrate
        endpoint += step_length #increment distance by 10pc
        Sigfunct = lambda r: dAv_dr(r, l_rad, b_rad) #so dAv_dr will intgrate correctly
        Sigma, err = integrate.quad(Sigfunct,startpoint,endpoint) #get magnitude loss due to extinction
        blocked += Sigma #add new light extinction to blocked

        distmod = 5*np.log10(endpoint*1000/10) #get magnitude loss due to distance

        newapp = distmod + blocked + absmag #apparent mag is abs mag + extinction + distance modulus
    maxvis = endpoint #I did this for intuitive purposes, it's largely redundant on the next line
    return maxvis


lcoor = np.linspace(0, l_max, n_l) # array of longitude coordinates
bcoor = np.linspace(0, b_max, n_b) # array of latitude coordinates

vals = np.zeros((len(bcoor), len(lcoor))) #set up space for values



for i in range(n_b): #calculate max disance for each point on plot
    for j in range(n_l):
        vals[i][j] = distfinder(lcoor[j], bcoor[i])


#Step 2: getting probability for each distance
def dPsn_drdOmega(radius):
    "integrand of supernova probability per solid angle on sky"
    "dN_sn/dr dOmega = r^2 q_sn(R,z) dr"
    "where R = R(r,l,b) and z = z(r,b)"

    global l_rad, b_rad
    global R_sun

    z = radius * np.sin(b_rad)
    r_par = radius * np.cos(b_rad)
    R = np.sqrt(r_par**2 - 2.*R_sun*r_par*np.cos(l_rad) + R_sun**2)

    dP_drdOmega = radius**2 * q_SN(R,z)

    return dP_drdOmega




def q_SN(R_gc,z_gc):
    "supernova density in Galactocentric cylindrical coordinates"

    global R_disk, h_disk
    global SN_dist

    if (SN_dist == 'Adams'):
        q_0 = 1./(4.*np.pi*R_disk**2*h_disk)
        q = q_0 * np.exp(-R_gc/R_disk) * np.exp(-np.abs(z_gc)/h_disk)

    elif (SN_dist == 'spherical'):
        q_0 = 1./(8.*np.pi*R_disk**3)
        r_gc = np.sqrt(R_gc**2 + z_gc**2)
        q = np.exp(-r_gc/R_disk)

    elif (SN_dist == 'thindisk'):
        q_0 = 1./(8.*np.pi*R_disk*3)
        r_gc = np.sqrt(R_gc**2 + z_gc**2)
        q = np.exp(-r_gc/R_disk)

    elif (SN_dist == 'Green'):
        alpha = 1.09
        beta = 3.87
        R_sun = 8.5 # kpc
        R_sn = 295
        R_0 = R_sun / beta # kpc
        h = 0.095 # kpc
        r_gc = np.sqrt(R_gc**2 + z_gc**2)
        q = (R_sn/(4*np.pi*gamma(alpha+2)*beta**alpha*R_0**2*h))*((abs(r_gc)/R_0)**alpha)*np.exp(-abs(r_gc)/R_0)*np.exp(-abs(z_gc)/h)

    else:
        q = 0

    return q

if (SN_type == 'Ia'):
    h_disk = h_ia
    R_disk = R_ia # kpc
    b_max_deg = 10.

    labtext = "Type Ia Supernovae"

    if (SN_dist == 'thindisk'):
        h_disk = 0.350 # kpc

elif (SN_type == 'CC'):
    h_disk = h_cc
    R_disk = R_cc
    b_max_deg = 10.

    labtext = "Core-Collapse Supernovae"

    if (SN_dist == 'thindisk'):
        h_disk = 0.350 # kpc

else:
    print ("bad SN type!")

#Set up array to hold dP/dOmega
#Using lcoor and bcoor from distance calculator for coordinates
dP_dOmega = np.zeros_like(vals)


P_sum = 0. #total probability sum

for i in range(n_b): #calculate max disance for each point on plot
    for j in range(n_l):
        b_rad = math.radians(bcoor[i])
        l_rad = math.radians(lcoor[j])
        dP_dOmega[i][j], err = integrate.quad(dPsn_drdOmega,0.,vals[i][j])
        P_sum += dP_dOmega[i][j]*np.cos(b_rad)

P_sum = 4.*P_sum*(b_max_deg/(n_b-1.))*(l_max_deg/(n_l-1.))*(np.pi/180.)**2

print(P_sum)
plt.contourf(lcoor, bcoor, dP_dOmega, cmap = 'rainbow') #plot quadrant 1
plt.contourf(-1*lcoor, bcoor, dP_dOmega, cmap = 'rainbow') #quadrant 2
plt.contourf(lcoor, -1*bcoor, dP_dOmega, cmap = 'rainbow') #quadrant 4
plt.contourf(-1*lcoor, -1*bcoor, dP_dOmega, cmap = 'rainbow') #quadrant 3
plt.xlim(-150, 150)
plt.ylim(-20, 20)
plt.colorbar()
plt.show()