import datetime 
import tensorflow as tf

import tensorflow_datasets as tfds
#from tensorflow_examples.models.pix2pix import pix2pix as p2p 
from  pix2pix import unet_generator, discriminator 
import os
import time
import numpy as np 

class CycleGAN(object):
    """CycleGAN  class.

      Args:
        epochs: Number of epochs.
        enable_function: If true, train step is decorated with tf.function.
        buffer_size: Shuffle buffer size..
        batch_size: Batch size.
    """

    def __init__(self, epochs, enable_function, pretrained =False, 
                workdir ='./', sigma= 0.3):
        self.checkpoint_path = f"{workdir}/checkpoints/"
        self.workdir = workdir 
        self.epochs = epochs
        self.enable_function = enable_function
        self.lambda_value = 10
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.sigma_noise= sigma 
        
        self.generator_g = unet_generator( output_channels=2,input_channels =2 ,  norm_type='instancenorm')
        self.generator_f = unet_generator(output_channels=2,input_channels =2  , norm_type='instancenorm')
        self.discriminator_x =    discriminator(norm_type='instancenorm', input_ch = 2, target=False)
        self.discriminator_y = discriminator(norm_type='instancenorm',input_ch=2,  target=False)
        self.checkpoint = tf.train.Checkpoint(generator_g=self.generator_g,
                           generator_f=self.generator_f,
                           discriminator_x=self.discriminator_x,
                           discriminator_y=self.discriminator_y,
                           generator_g_optimizer=self.generator_g_optimizer,
                           generator_f_optimizer=self.generator_f_optimizer,
                           discriminator_x_optimizer=self.discriminator_x_optimizer,
                           discriminator_y_optimizer=self.discriminator_y_optimizer)
        
        self. ckpt_manager = tf.train.CheckpointManager(self.checkpoint,self.checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint and pretrained :
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print ( self.ckpt_manager.latest_checkpoint ,'Latest checkpoint restored!!')
        
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(
            tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(
            disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss*0.5

    def generator_loss(self, disc_generated_output, gen_output, target):
        
        gan_loss = self.loss_object(tf.ones_like(
            disc_generated_output), disc_generated_output)
        return gan_loss 
    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.lambda_value  * loss1
        # mean absolute error

    def identity_loss(self,real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lambda_value * 0.5 * loss

    def train_step(self, real_x, real_y):
        """One train step over the generator and discriminator model.

        Args:
          input_image: Input Image.
          target_image: Target image.

        Returns:
          generator loss, discriminator loss.
        """
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            noise_x = tf.random.normal(stddev=self.sigma_noise, shape=real_x.shape )
            #uniform(shape=real_x.shape ,maxval=.6)
            noise_y = tf.random.normal(stddev=self.sigma_noise, shape=real_y.shape )
            #uniform(shape=real_y.shape ,maxval=.6)
            tf.math.add (noise_x, real_x, name='Adding_noiseX' ) 
            tf.math.add (noise_y, real_y, name='Adding_noiseY' ) 

            fake_y = self.generator_g(real_x, training=True) 
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)
            #print(cycled_y.shape, cycled_x.shape ) 
            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss =self. generator_loss(disc_fake_y, fake_y, real_y)
            gen_f_loss = self.generator_loss(disc_fake_x, fake_x, real_x )

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                            self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                            self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                               self. generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                   self. discriminator_y.trainable_variables))
        losses = (total_gen_g_loss , total_gen_f_loss, disc_x_loss, disc_y_loss)

        return [l for l in losses ]
 

    def train(self, BATCH_SIZE,xtrain, ytrain    ):
        """Train the GAN for x number of epochs.
 
        """ 
        if self.enable_function:
            self.train_step = tf.function(self.train_step)


        nbatch = xtrain.shape[0]//BATCH_SIZE
        self.losses_arr = np.empty((4, self.epochs ) )
        for epoch in range(self.epochs):
            losses_val =0 
            start = time.perf_counter() 

            for n in range(nbatch-1):
                losses  = self.train_step(xtrain[n*BATCH_SIZE:(n+1)*BATCH_SIZE], 
                           ytrain[n*BATCH_SIZE:(n+1)*BATCH_SIZE])
                losses_val += np.array([li.numpy()  for li in losses  ]   ) 
                if n % 10 == 0: print ('.', end='')

            self.losses_arr[:,epoch]  = losses_val  / (nbatch-1 )

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                              time.perf_counter()-start))
            if (epoch ) % 50  == 0:    
                ckpt_save_path =self.ckpt_manager.save()
                print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        

        
        
    def predict (self, xtest, ytest ): 
        self.ypred  = self.generator_g(xtest+np.random.normal ( scale=self.sigma_noise ,  size= xtest.shape))
        self.xpred  = self. generator_f(ytest+np.random.normal ( scale=self.sigma_noise, size= ytest.shape))

    def predict_COmap (self, xtest, fname   ): 
        self.ypred  = self.generator_g(xtest+np.random.normal ( scale=self.sigma_noise ,  size= xtest.shape))
        np.savez(fname ,  ypred= self.ypred   )
        
    def save_predictions(self):
        np.savez(f'{self.workdir}/CO_cyclegan_predictions.npz',  ypred= self.ypred , xpred=self.xpred )
    
    def save_losses (self ):
        now = datetime.datetime.now ()
        np.savez(f"{self.workdir}/losses_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}.npz" , 
                 losses_g=self.losses_arr[0,:],
                 losses_f=self.losses_arr[1,:],
                 losses_discx=self.losses_arr[2,:],
                 losses_discy =self.losses_arr[3,:])    
