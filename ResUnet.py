import pylab as pl

import numpy as np
import tensorflow as tf
from keras import layers
import keras.backend as K
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,UpSampling2D, BatchNormalization, Conv2D,Flatten,Reshape, ZeroPadding2D, Dropout
from keras.models import Model, model_from_json
from keras.initializers import glorot_uniform
from keras.optimizers import RMSprop
import os



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
def conv_layer_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- integer, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    conv_id =   block + str(stage)
    X= Dropout(rate=0.3, seed=stage, name='dropout_'+conv_id ) (X)
    if s==1:
        X = Conv2D( filters  ,  (f,f)  , padding='same',
                   kernel_initializer = glorot_uniform(seed=0), name='conv_'+conv_id)(X)
    elif s==2:
        X = Conv2D( filters  ,  (f,f) , strides=(s,s) , padding='same', name='conv_stride_'+conv_id)(X)

    X=Activation('selu' ,  name='seLU_'+conv_id)(X)

    X = BatchNormalization(axis=3, name='BN_'+conv_id) (X)

    return X

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

class ResUNet:

    def __init__(self, output_directory='./', img_size=64,
                    epochs=500, batch_size=128, verbose=True,
                    pretrained= False , nchans=1 , network_name='ResUNet' ):
        self.img_size = img_size
        self.nchannels =nchans
        self.model_directory = output_directory
        self.epochs=epochs
        self.verbose =verbose
        self.batch_size=batch_size
        self.pretrained= pretrained
        self._name= network_name

        if self.verbose :
            print(f"{self._name} running on   {self.img_size, self.img_size} images" )

    def load_model (self):
        json_file = open(self.model_directory+f'/models/{self._name}_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.compile(loss="mean_squared_error" ,
                        optimizer = "Adam", metrics=['accuracy'])
        # load weights into new model
        self.model.load_weights(self.model_directory+f"/models/{self._name}_model.h5")
        if self.verbose: print("Loaded model from disk")




    def build_resunet(self):
        f=5
        X_input =  Input(  (self.img_size,self.img_size, self.nchannels ) )
        #encoder
        X= conv_layer_block(X_input,f, filters=64, stage=1, block='enc_', s=1) #64 x128
        Xin=X

        X= conv_layer_block(X ,f, filters=64, stage=2, block='enc_', s=1)
        X= conv_layer_block(X ,f, filters=64, stage=3, block='enc_', s=1) #skip-conn 1

        Xout1=X #64x128 add w/ input of 11-12
        X = Add( )([Xin, X ]) #input layer 2  + input layer 4

        Xin=X #64x128

        X= conv_layer_block(X ,f, filters=128, stage=4, block='enc_', s=2) #128x64
        X= conv_layer_block(X ,f, filters=128, stage=5, block='enc_', s=1)

        Xin=Conv2D(128  ,  (f,f) , strides=(2,2) , padding='same', name='conv_input_1') (Xin) #128x 64
        X = Add( )([Xin, X ]) #input layer 4  + input layer 6
        Xin=X

        X= conv_layer_block(X ,f, filters=128, stage=6, block='enc_', s=1) #skip-conn 2
        Xout2=X  #combine w/ input 8-9

        X= conv_layer_block(X ,f, filters=256, stage=7, block='enc_', s=2) #256 x 32
        Xin=Conv2D(256  ,  (f,f) , strides=(2,2) , padding='same', name='conv_input_2')(Xin) #256x 32

        X=  conv_layer_block(X ,f, filters=128, stage=8, block='dec_', s=1) #128 x32
        X = UpSampling2D((2,2),interpolation='nearest', name='upsample_1')(X) #128x 64
        #skip-conn2
        X = Add( )([Xout2, X ])

        Xin=X

        X= conv_layer_block(X ,f, filters=128, stage=9, block='dec_', s=1)
        X= conv_layer_block(X ,f, filters=128, stage=10, block='dec_', s=1)
        X = Add( )([Xin, X ]) #input layer 9  + input layer 11
        Xin=X

        X= conv_layer_block(X ,f, filters=64, stage=11, block='dec_', s=1) #64x64
        X = UpSampling2D((2,2),interpolation='nearest', name='upsample_2' )(X) # 64 x 128

        #skip-conn 1
        X = Add( )([Xout1, X ])

        X= conv_layer_block(X ,f, filters=64, stage=12, block='dec_', s=1)

        Xin= Conv2D(64  ,  (f,f) ,   padding='same', name='conv_input_3') (Xin)
        Xin = UpSampling2D((2,2),interpolation='nearest', name='upsample_3' )(Xin) # 64 x 128

        X = Add( )([Xin, X ]) #input layer 11  + input layer 13
        Xin=X
        #skip-conn 1
        X= conv_layer_block(X ,f, filters=64, stage=13, block='dec_', s=1)
        X= conv_layer_block(X ,f, filters=64, stage=14, block='dec_', s=1)
        X = Add( )([Xin, X ]) #input layer 13 + input layer 15

        Xin= Conv2D(1  ,  (f,f) ,  padding='same', name='conv_input_4') (X )


        X= conv_layer_block(X ,f, filters=1, stage=15, block='dec_', s=1) #1x128
        X= conv_layer_block(X ,f, filters=1, stage=16, block='dec_', s=1) #1x128
        X = Add( )([Xin, X ]) #input layer 15  + last layer

        self.model = Model(inputs = X_input, outputs = X, name='ResUNet')
        self.model.compile(loss='mean_squared_error',
                        optimizer = "Adam",
                        metrics=['accuracy'])


    def train(self, x_train  ,y_train, x_val, y_val ):

        if self.pretrained:
            self.load_model()
        else:
            self.build_resunet( )

        if self.verbose :
            self.model.summary()

        training= self.model.fit(x_train , y_train , epochs=self.epochs ,
                                        batch_size=self.batch_size ,
                                     shuffle=True, verbose=2 ,
                                     validation_data=(x_val ,y_val  ))
        scores = self.model.evaluate(x_train, y_train, verbose=self.verbose)
        if self.verbose :
            print( f"{self.model.metrics_names[1]} :  {scores[1]}" )
            print( f"{self.model.metrics_names[0]} :  {scores[0]}" )
        save_path = self.model_directory + "/models/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for key in training.history.keys() :
            np.save(save_path + f'/{key}_{self._name}{self.epochs}.npy', np.array (training.history[key]))
        model_json=  self.model.to_json()
        with open(save_path+f'{self._name}_model.json', "w")  as json_file:
            json_file.write(model_json)
        self.model.save_weights(save_path+f"{self._name}_model.h5")
        if self.verbose :
            print("saved model to disk")

    def predict(self, x_test ):
        print(f'running predictions on {x_test.shape}  images ')

        return  self.model.predict(x_test)
