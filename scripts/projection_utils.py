import healpy as hp 
import pylab as pl 
import astropy 
from astropy import units as u 
import collections 
import reproject
import numpy as np
import astropy.io.fits as fits

import argparse 
import time     
import warnings 
warnings.filterwarnings("ignore")


def set_header(ra,dec, size_patch ,Npix=128 ):
    hdr = fits.Header()
    hdr.set('SIMPLE' , 'T')
    hdr.set('BITPIX' , -32)
    hdr.set('NAXIS'  ,  2)
    hdr.set('NAXIS1' ,  Npix)
    hdr.set('NAXIS2' ,  Npix )
    hdr.set('CRVAL1' ,  ra)
    hdr.set('CRVAL2' ,  dec)
    hdr.set('CRPIX1' ,  Npix/2. +.5)
    hdr.set('CRPIX2' ,  Npix/2. +.5 )
    hdr.set('CD1_1'  , size_patch )
    hdr.set('CD2_2'  , -size_patch )
    hdr.set('CD2_1'  ,  0.0000000)
    hdr.set('CD1_2'  , -0.0000000)
    hdr.set('CTYPE1'  , 'RA---ZEA')
    hdr.set('CTYPE2'  , 'DEC--ZEA')
    hdr.set('CUNIT1'  , 'deg')
    hdr.set('CUNIT2'  , 'deg')
    hdr.set('COORDSYS','icrs')
    return hdr

def h2f(hmap,target_header,coord_in='C'):
    #project healpix -> flatsky
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

def f2h(flat,target_header,nside,coord_in='C'):
    #project flatsky->healpix
    pr,footprint = reproject.reproject_to_healpix(
    (flat, target_header),coord_system_out='C', nside=nside ,
    order='nearest-neighbor', nested=False)
    return pr, footprint
    
def project2hpx (  flatmap , nside ,phi, theta ,sizedeg  ) : 
    Np =flatmap.shape[0]
    ij2coords =lambda i,j: ( (i -Np /2 )*sizedeg +  theta ,(j -Np /2 )*sizedeg +  phi )
    i_p , j_p = pl.meshgrid(pl.arange(Np, dtype=pl.int_ ) , pl.arange(Np, dtype=pl.int_))
    phip ,thetap =ij2coords ( j_p .ravel (),i_p.ravel() )
    assert phip.shape== flatmap.flatten().shape
    hpx_pix= hp.ang2pix( nside =nside , phi= phip, theta= thetap, lonlat=True )
    newmap =pl.zeros(hp.nside2npix(nside)) 
    newmap[hpx_pix] = flatmap.flatten() 
    
    return newmap 

def get_lonlat( size_patch ,  overlap ): 
    Nlon = np.int_( np.ceil(360.*u.deg / (size_patch  -overlap  )  ).value  )   
    Nlat =np.int_  (np.ceil( 180. *u.deg/( size_patch-overlap )  ).value )    +1
 
    rLon =(360.*u.deg % (size_patch  -overlap  )).value; rLat= (180. *u.deg%( size_patch-overlap ) ) .value 

    offset_lon = 0 
    offset_lat= -90 
    lat_array= np.zeros(np.int_(Nlat) )  
    lon_array= np.zeros(np.int_(Nlon) )
    lat_array[:Nlat] = [ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat) ]
    lon_array[:Nlon ] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon ) ]
    
    if rLon==0 and rLat==0:
        lat_array[:Nlat] = [ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat) ]
        lon_array[:Nlon ] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon ) ]
    else : 
        lat_array[:Nlat-1] = [ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat-1) ]
        lon_array[:Nlon -1] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon-1 ) ]
        lat_array[Nlat-1] =  lat_array[-2 ]+ rLat 
        lon_array[Nlon-1] =  lon_array[-2 ]+ rLon  
        
    lon , lat  =pl.meshgrid(lon_array,lat_array)
    return   lon.ravel(),lat.ravel()  


def get_lonlat_adaptive  ( size_patch ,  overLon, overLat  ): 
    Nlon = np.int_( np.ceil(360.*u.deg / (size_patch  -overLon  )  ).value  )   
    Nlat =np.int_  (np.ceil( 180. *u.deg/( size_patch-overLat )  ).value )
    if Nlat %2 ==0 :
        Nlat+= 1 
    offset_lon = 0 
    offset_lat= -90 
    lat_array= np.zeros(np.int_(Nlat) )  
    lon_array= np.zeros(np.int_(Nlon) )
    
    lat_array[:Nlat//2   ] =[ offset_lat + ((size_patch ).value -overLat.value) *i for i in range(  Nlat//2  )  ] 
    lat_array[Nlat//2+1 : ] =[ -offset_lat -  ((size_patch ).value -overLat.value) *i for i in range(  Nlat//2 ) ] [::-1]
    lat_array[Nlat//2] =  0
    lon_array[:Nlon ] = [offset_lon + ((size_patch ).value -overLon.value )*i for i in range( Nlon ) ]
    Nloneff = np.int_( np.cos(np.radians(lat_array))*Nlon ) 
    Nloneff[0]=5; Nloneff[-1] =5
    jumps = (np.int_(np.ceil(Nlon/Nloneff ) -1)  )
    jumps[1] -=1 
    jumps[-2] -=1 
    jumps [Nlat//2] =1 
    lon , lat  =pl.meshgrid(lon_array,lat_array)
    lonj=[] 
    latj =[] 
    for kk in range(Nlat ): 
        lonj.append(lon[kk,::jumps[kk] ])
        latj.append(lat[kk,::jumps[kk]] )
    lonj=np.concatenate(lonj); latj=np.concatenate(latj)
    return lonj, latj 
    

def make_mosaic_from_healpix( hpxmap , Npix, pixel_size , overlap=5 *u.deg , adaptive=False  ):
    patches = []
    size_patch = pixel_size *Npix  
    if adaptive: 
        lon,lat = get_lonlat_adaptive(size_patch , overLat=overlap, overLon=overlap )
        
    else:
        lon,lat = get_lonlat(size_patch , overlap )
        
    for  phi,theta in zip (lon , lat   ):
        header = set_header(phi, theta, pixel_size.value , Npix)
        patches.append(h2f(hpxmap , header))
         
    patches = np.array( patches )
    return patches, lon , lat 

 

def reproject2fullsky ( tiles, lon, lat , 
                        nside_out, pixel_size, Npix, apodization_file=None , 
                       verbose=False, comm=None   ):
    if comm is None :
        rank =0 
    else :
        rank =comm.Get_rank() 
        
    newmap =pl.zeros(hp.nside2npix((nside_out)) )
    weightsmap = pl.zeros_like(newmap )
    flatmap = pl.ones ((Npix,Npix))
    sizedeg= pixel_size.to(u.deg)
    development= 0 
    s=time.perf_counter() 
    try: 
        apoflat = pl.load(apodization_file )
    except TypeError: 
        apoflat=flatmap
    
    for p, phi, theta  in zip ( tiles , lon  ,lat  )  : 
        header = set_header(phi, theta , sizedeg.value ,Npix  ) 
        
        tmpmap,fp= f2h (p*apoflat ,header ,nside_out  ) 
        tmpmap [ pl.ma.masked_invalid(tmpmap).mask  ]=0 
        
        newmap+=tmpmap 
        weightsmap +=fp 

        development +=1 
        
        #if development %10==0 and verbose :
            #if  rank ==0 : print(f"{development } out of {len(tiles)} in {e-s} sec" )
           
    e=time.perf_counter() 
    if  rank ==0 : print(f"   {len(tiles)} projected  in {e-s} sec" )

    return newmap, weightsmap 