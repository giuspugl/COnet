
import numpy as np 

import argparse 

import torch 

def augment_dataset (arr ) :
    # augment training   set by flipping the axis
    flipped1 =np.array([  np.flip( p, axis=1)  for p in arr ] )

    flipped0 =np.array([  np.flip( p, axis=0)  for p in arr ] )
    return ( np.concatenate((arr,  flipped0,  flipped1  ) ,axis=0 ) )


def minmaxrescale(x, a=0, b=1):
    """
    Performs  a MinMax Rescaling on an array `x` to a generic range :math:`[a,b]`.
    """
    xresc = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
    return xresc


def split_trainvaltest_sets(xraw):
    nstamps=xraw.shape[0 ]
    npix =xraw.shape[1]
    nchans=xraw.shape[-1]

    ntrains= int (nstamps *  4./5.)
    nvals =   int (nstamps * 1./10.)
    ntests =    int (nstamps * 1./10.)
    return (   xraw[ :ntrains]  ,  xraw[  ntrains:ntrains + nvals] ,  xraw[ -ntests:]  )


def split_without_overlap(xraw, longitudemask ):
    training = longitudemask [0]
    validation = longitudemask [1]
    test = longitudemask [2]
    return xraw[training], xraw[validation], xraw[test ]

def preprocess_data(  arr,  split_wo_overlap=False, longitudemask =None , rescale = True  ):
    if rescale :
        if arr.shape[-1]>1:
            for i in range(arr.shape[0]):
                for k in range(arr.shape[-1]) :
                    arr[i,:,:,k]=minmaxrescale(arr[i,:,:,k], a=-1,b=1 )
        else:
            for i in range(arr.shape[0]):
                arr[i ]=minmaxrescale(arr[i ], a=-1,b=1 )
    if  split_wo_overlap and longitudemask is not None  :
        xtrain, xval,xtest =split_without_overlap  (arr , longitudemask  )
    elif  split_wo_overlap and longitudemask is   None  :
        raise ValueError (f'longitudemask is set to {longitudemask}, must be an array to split the data ')
    else :
        xtrain, xval,xtest =split_trainvaltest_sets(arr  )

    return xtrain, xval,xtest


    
def main(args):


    workdir=args.workdir
    dust = np.load(f'{workdir}/COM_CompMap_Dust-GNILC-F857_2048_R2.00_15amin_training.npz')['patches']
    nh = np.log10(np.load(f'{workdir}/NHI_HI4Pi_16amin_nside2048_inpainted_training.npz')['patches']) 
    co10 = np.load(f'{workdir}/CO10type2_15amin_nside2048_training.npz')['patches']
    co21 = np.load(f'{workdir}/CO21type2_15amin_nside2048_training.npz')['patches']

    
    Xin   = np.float32(np.stack([dust,nh ], axis=-1 ) ) 
    Xout = np.float32(np.stack([co10,co21  ], axis=-1 )) 
    if args. longitudinal_split :
        masklongitudes = np.load(workdir + args.longitudinal_mask )
    else:
        masklongitudes=None

    xtrain ,xval, xtest =  preprocess_data ( arr=Xin ,split_wo_overlap=args.longitudinal_split  , longitudemask =masklongitudes  )
    ytrain ,yval, ytest =  preprocess_data ( arr=Xout  ,split_wo_overlap=args.longitudinal_split  , longitudemask =masklongitudes , rescale=args.rescale_outputs   )
    if args.augment_trainingset :
        xtrain = augment_dataset (xtrain)
        xval = augment_dataset (xval )
        ytrain = augment_dataset (ytrain)
        yval = augment_dataset (yval )
        xtest = augment_dataset (xtest )
        ytest = augment_dataset (ytest )
        

    #fimport pdb ; pdb.set_trace()

    xtrain = np.vstack((xtrain,xval)) 
    ytrain = np.vstack((ytrain,yval)) 
    
    # 2. Convert to PyTorch Tensor
    # torch.from_numpy() creates a tensor that shares memory with the array (efficient)
    
    tensor_data_A = torch.from_numpy(xtrain).float().permute(0,3,1,2) # .float() is usually safer for NN training
    tensor_data_B = torch.from_numpy(ytrain).float().permute(0,3,1,2) # .float() is usually safer for NN training
    print(tensor_data_A.shape, tensor_data_B.shape)

    # 3. Save to .pt file
    torch.save(tensor_data_A, f"{workdir}/fileA.pt")
    torch.save(tensor_data_B, f"{workdir}/fileB.pt")



if __name__=="__main__":
    parser = argparse.ArgumentParser( description="  " )

    parser.add_argument("--workdir" ,  default='./'  )
    parser.add_argument("--longitudinal-split" ,  action='store_true', default=False    )
    parser.add_argument("--longitudinal-mask" ,   type=str  ,default='co_ext_longitudinal_splitmask.npy'  )
    parser.add_argument("--verbose" , action='store_true', default=False  )
    parser.add_argument("--augment-trainingset" , action='store_true', default=False  )
    parser.add_argument("--rescale-outputs" , action='store_true', default=False  )

    args = parser.parse_args()


    main( args)
    