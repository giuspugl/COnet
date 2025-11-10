from CycleGAN import CycleGAN 
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ResUnet import ResUNet , preprocess_data  , augment_dataset
import argparse 




def main(args):

    BATCH_SIZE = args.batch_size 
    EPOCHS=args.epochs 

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
        


    xtrain = np.vstack((xtrain,xval))# we don't need validation for GANs
    ytrain = np.vstack((ytrain,yval))# we don't need validation for GANs

    
    cgan  = CycleGAN( epochs= EPOCHS, enable_function=True ,  
                 pretrained =args.pretrained , workdir=workdir, sigma= args.sigmanoise ) 

    

    cgan. train(BATCH_SIZE=BATCH_SIZE, 
                    xtrain=xtrain,
                    ytrain=ytrain ) 
    cgan.save_losses() 
    
    cgan.predict(xtest, ytest) 
    
    cgan. save_predictions() 




if __name__=="__main__":
    parser = argparse.ArgumentParser( description="  " )

    parser.add_argument("--epochs" , type=np.int_ )
    parser.add_argument("--batch_size" , type=np.int_ )
    parser.add_argument("--pretrained" , action='store_true', default=False  )
    parser.add_argument("--workdir" ,  default='./'  )
    parser.add_argument("--longitudinal-split" ,  action='store_true', default=False    )
    parser.add_argument("--longitudinal-mask" ,   type=str  ,default='co_ext_longitudinal_splitmask.npy'  )
    parser.add_argument("--verbose" , action='store_true', default=False  )
    parser.add_argument("--augment-trainingset" , action='store_true', default=False  )
    parser.add_argument("--rescale-outputs" , action='store_true', default=False  )
    parser.add_argument("--sigmanoise" , type=float  )
    

    args = parser.parse_args()


    main( args)

    """
    --epochs 500  --batch_size 128 --workdir /pscratch/sd/g/giuspugl/workstation/CO_network/extending_CO/ --pretrained --longitudinal-split --augment-trainingset --rescale-outputs
    """