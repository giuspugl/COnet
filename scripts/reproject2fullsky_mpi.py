import healpy as hp
import pylab as pl
import astropy
from astropy import units as u
import collections
import reproject
import numpy as np
import astropy.io.fits as fits
from mpi4py import MPI
import os.path
from os import path
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

from projection_utils import (
                    get_lonlat, get_lonlat_adaptive,
                     reproject2fullsky,  make_mosaic_from_healpix  )
def main(args):
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    nprocs  = comm.Get_size()

    Npix= pl.int_(args.npix )
    pixel_size = args.pixelsize  *u.arcmin
    overlap = args.overlap *u.deg
    nside_in=pl.int_(args.nside) 

    hpxsize  = hp.nside2resol(nside_in, arcmin=True ) *u.arcmin
    nside_out = pl.int_( nside_in )

    if args.flat2hpx :
        """
        I assume that each set of square patches encode just T or Q or U maps,
        if you want a TQU .fits map  hpx reprojected map needs to further postprocessing

        """
        if args.verbose and rank==0 : print(f"reading patches from {args.flat_projection}")

        if args.line =="10":
            line =0
        elif args.line=="21":
            line=1

        #patches    = pl.load(args.flat_projection , allow_pickle=True)['patches']
        patches    = pl.load(args.flat_projection  )['ypred'][:,:,:,line  ]
        print(patches.shape )
        size_patch = pixel_size.to(u.deg) *Npix

        if args.adaptive_reprojection :
            lon,lat =get_lonlat_adaptive(size_patch , overLat= overlap ,overLon=overlap  )

        else:
            lon,lat =get_lonlat(size_patch , overlap )
        localsize = np.int_(lon.size /nprocs)
        remainder = lon.size % nprocs
        if (rank < remainder) :
        #  The first 'remainder' ranks get 'count + 1' tasks each
            start = np.int_(rank * (localsize  + 1))
            stop =np.int_(  start + localsize +1  )
        else:
        # The remaining 'size - remainder' ranks get 'count' task each
            start = np.int_(rank * localsize + remainder  )
            stop =np.int_(  start + (localsize  )   )

        filename = args.flat_projection .replace( '.npz',f'_{args.line}.fits')
        if args.verbose and rank==0 :
            print("reprojecting back to HPX")
            print (f"files will be stored in {filename}")

        s= time.perf_counter()
        
        #print(rank,   patches[start:stop].shape, lon [start:stop].shape, lat [start:stop].shape )
        newmap, weightsmap = reproject2fullsky(  tiles=patches[start:stop], lon=lon [start:stop], lat=lat [start:stop],
                                            nside_out=nside_out, pixel_size=pixel_size ,
                                            apodization_file =args.apodization_file  ,
                                             Npix = Npix, verbose=True ,comm=comm
                                                )
        reducedmap=  np.zeros_like(newmap)
        comm.Reduce(newmap , reducedmap, op=MPI.SUM)
        reducedwmap=  np.zeros_like(newmap)
        comm.Reduce(weightsmap , reducedwmap, op=MPI.SUM)
        comm.Barrier()
        e= time.perf_counter ()
        if args.apodization_file is not None:
            if path.exists( args.apodization_file .replace('.npy', '.fits')):

                reducedapomap= hp.read_map(args.apodization_file .replace('.npy', '.fits'), verbose= args.verbose)
                if args.verbose and rank==0:
                    print('Apodized  map already saved ')
                    hp.write_map(filename  , [reducedmap /reducedapomap  , reducedmap, reducedwmap], overwrite=True  )

            else :
                if args.verbose  and rank==0: print("reprojecting apomask")
                apomap, _ = reproject2fullsky(  tiles=np.ones_like(patches) [start:stop] , lon=lon[start:stop] ,       lat=lat[start:stop] ,
                                                nside_out=nside_out, pixel_size=pixel_size ,
                                                apodization_file =args.apodization_file  ,
                                                Npix = Npix, verbose=True, comm=comm                )


                reducedapomap=  np.zeros_like(newmap)
                comm.Reduce(apomap , reducedapomap, op=MPI.SUM)
                if rank==0 :
                    hp.write_map( args.apodization_file .replace('.npy', '.fits') , reducedapomap  , overwrite=True  )
                    
                    hp.write_map(filename  , [reducedmap /reducedapomap  , reducedmap, reducedwmap], overwrite=True  )
        else:
            if rank==0: hp.write_map(filename  , [reducedmap /reducedwmap   , reducedmap, reducedwmap], overwrite=True  )

        comm.Barrier()

        if args.verbose and rank==0 : print(f"process took {e-s} sec")

    elif args.hpx2flat :
        if args.has_polarization :
            inputmap = hp.read_map(args.hpxmap, verbose =args.verbose, field=[0,1,2] )
            stringmap ='TQU'
        else:
            stringmap='T'
            inputmap = [ hp.read_map(args.hpxmap, verbose =args.verbose )    ]

        filename  = args.hpxmap.replace('.fits','.npz')
        assert len(stringmap)== len(inputmap )
        assert  nside_in == hp.get_nside(inputmap)

        if args.verbose and rank==0  :
            print(f"Making square tile patches {pixel_size.to(u.deg) *Npix } x {pixel_size.to(u.deg) *Npix } from {args.hpxmap}")
            print (f"files will be stored in {filename}")
        for   imap,maptype   in zip(inputmap, stringmap ) :

            s= time.pref_counter()
            patches, lon, lat = make_mosaic_from_healpix(  imap, Npix, pixel_size.to(u.deg) , overlap=  overlap ,adaptive=args.adaptive_reprojection    )
            e= time.pref_counter ()

            pl.savez(filename.replace('.npz',f'_{maptype}.npz') , patches= patches , lon=lon, lat=lat   )
            if args.verbose and rank==0  : print(f"process took {e-s} sec ")

    comm.Disconnect

    return

if __name__=="__main__":
    parser = argparse.ArgumentParser( description="" )
    parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked' )
    parser.add_argument("--pixelsize", help = 'pixel size in arcminutes of the input map',type=float,   default = 3.75 )
    parser.add_argument("--npix", help='size of patches',type=int , default = 320,  )
    parser.add_argument("--nside", help='nside of output map ',type=int , default = 2048,  )
    parser.add_argument("--overlap", help='partial patch overlap in deg',type=float, default=5, )
    parser.add_argument("--flat2hpx", action="store_true" , default=False )
    parser.add_argument("--hpx2flat", action="store_true" , default=False )
    parser.add_argument("--verbose", action="store_true" , default=False  )
    parser.add_argument("--flat-projection",  help='path to the file with list of patches  ', default ='' )
    parser.add_argument("--has-polarization",  help='include polarization', default =False, action="store_true"  )
    parser.add_argument("--apodization-file",  help='path of the apodization mask', default =None   )
    parser.add_argument("--adaptive-reprojection",  help='adaptive reprojection', default =False, action="store_true"  )
    parser.add_argument("--line",  help='which CO line to reproject, one string between 10 or 21   ', default ="10",   )
    args = parser.parse_args()
    main( args)
