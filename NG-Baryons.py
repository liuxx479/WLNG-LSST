from scipy import *
import numpy as np
import astropy.units as u
from lenstools import ConvergenceMap
from emcee.utils import MPIPool
import sys

i=int(sys.argv[1])
Osato_dir = lambda i: '/scratch/02977/jialiu/OsatoSims/%s_lensdat/'%(['DM', 'FE', 'BA'][i])

def NGstats_gen(num, i=0):
    folder = Osato_dir(i)+'run%02d/'%(num)
    print folder

    conv_map = load(folder+'Combined.npy')[:,0].reshape(4096,4096)    
    conv_map = ConvergenceMap(conv_map, angle=5.0*u.deg)

    ## smooth the conv_map
    conv_smooth=conv_map.smooth(2.0*u.arcmin,kind="gaussianFFT",inplace=0)
    istd = 0.02#std(conv_smooth)
    kappabins = np.linspace(-3.,5., 51)*istd

    ## 1.peak counts
    ## if norm=1, assume kappabins to be in unit of noise
    #height, positions = conv_smooth.locatePeaks(kappabins,norm=0) 
    nu, peaks = conv_smooth.peakCount(kappabins, norm=False)

    ## 2.troughs
    conv_neg = ConvergenceMap(-conv_smooth.data, angle=5.0*u.deg)
    nu_neg, troughs = conv_neg.peakCount(kappabins, norm=False)

    ## 3.MFs
    # returns: tuple -- (nu -- array, V0 -- array, V1 -- array, V2 -- array) nu are the bins midpoints and V are the Minkowski functionals
    ## MFs = [nu, v0, v1, v2]
    MFs = conv_smooth.minkowskiFunctionals(kappabins,norm=0)[1:]

    ## 4.PDF
    nu, pdf = conv_smooth.pdf(kappabins, norm=False)

    ## 5.moments
    ## output sigma0,sigma1,S0,S1,S2,K0,K1,K2,K3 for var0,var1,sk0,sk1,sk2,kur0,kur1,kur2,kur3
    moments = conv_smooth.moments(connected=False, dimensionless=False)

    return nu, peaks, troughs, MFs, pdf, moments

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
    
out = pool.map(NGstats_gen, range(100))
save('/scratch/02977/jialiu/OsatoSims/stats/NG_%s.npy'%(['DM', 'FE', 'BA'][i]), out)
print "DONE DONE DONE"

