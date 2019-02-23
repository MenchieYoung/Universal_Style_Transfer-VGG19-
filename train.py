"""
ECBM 4040 
Group YYZA
Group Members: Manqi Yang (my2577), Shiyun Yang (sy2797), Yizhi Zhang (yz3376)
"""

import tensorflow as tf
import numpy as np
from skimage.io import imsave,imshow,imread
from PIL import Image as pil_image
from Encoder import Vgg19
from Decoder import Decoder
from wct import wct_tf
from argparse import ArgumentParser


        
class Train_Model:
    """
    Train Model Class: 
    this class is used for decoder training
    """
    
    def __init__(self, target_layer=None, pretrained_path=None, max_iterator=None, checkpoint_path=None, 
                 tfrecord_path=None, batch_size=None):
        """
        Train Model Initialization:
        
        Inputs: 
        target_layer: indicate which decoder to train (decoder1 -> relu1, ..., decoder5 -> relu5)
        pretrained_path: path to pre-trained VGG19 weights (vgg19.npy)
        max_iterator: number of iterations for training
        checkpoint_path: path to save genertated model
        tfrecord_path: path to tfrecord file of training data
        batch_size: batch size for training
        """
        self.pretrained_path = pretrained_path
        self.target_layer = target_layer
        self.encoder = Vgg19(self.pretrained_path)
        self.max_iterator = max_iterator
        self.checkpoint_path = checkpoint_path
        self.tfrecord_path = tfrecord_path
        self.batch_size = batch_size
      
    
    def encoder_decoder(self,inputs):
        """
        encoder_decoder function:
        
        Outputs:
        encoded: feature map obtained by processing the original image with the encoder
        decoded: reconstructed image obtained by processing the original image with both the encoder and decoder
        decoded_encoded: feature map obtained by processing the reconstructed image with the encoder
        """
        encoded = self.encoder.encoder(inputs,self.target_layer)
        model=Decoder()
        decoded,_ = model.decoder(encoded,self.target_layer)
        decoded_encoded= self.encoder.encoder(decoded,self.target_layer)
        
        return encoded,decoded,decoded_encoded
    
    def train(self):
        """
        Train function: 
        trains the decoder specified
        """
        
        # save the pixel losses and feature losses during training so we can plot
        p_loss_list = []
        f_loss_list = []
        
        inputs = tf.placeholder('float',[None,224,224,3])
        outputs = tf.placeholder('float',[None,224,224,3])
        
        encoded,decoded,decoded_encoded = self.encoder_decoder(inputs)
        
        # compute pixel loss and feature loss: 
        # pixel loss = square L2 norm of Io-Ii, where Io is the reconstructed image and Ii is the original image
        # feature loss = square L2 norm of phi(Io)-phi(Ii), 
        # where phi(Io) is the feature map of the reconstructed image computed by VGG encoder
        # and phi(Ii) is the feature map of the original image
        pixel_loss = tf.losses.mean_squared_error(decoded,outputs)
        feature_loss = tf.losses.mean_squared_error(decoded_encoded,encoded)
        
        # total loss is computed as the sum of pixel loss and feature loss
        loss = pixel_loss+ feature_loss
        opt= tf.train.AdamOptimizer(0.0001).minimize(loss)
        
        # we use TensorFlow Record (binary format) for our training data, this gives better efficiency than processing data directly 
        tfrecords_filename =  self.tfrecord_path
        filename_queue = tf.train.string_input_producer([tfrecords_filename],num_epochs=100)

        reader = tf.TFRecordReader()  
        _, serialized_example = reader.read(filename_queue)

        feature2 = {'image_raw': tf.FixedLenFeature([], tf.string)} 
        features = tf.parse_single_example(serialized_example, features=feature2)  
        image = tf.decode_raw(features['image_raw'], tf.uint8) 
        image = tf.reshape(image,[224,224,3])   
        images = tf.train.shuffle_batch([image], batch_size=self.batch_size, capacity=30, min_after_dequeue=10)
        
        # GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config = config)as sess  :
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            # we use TensorFlow's RandomShuffleQueue to prepare training inputs
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(coord=coord)  

            saver = tf.train.Saver()

            # training iterations
            for i in range (self.max_iterator):
                batch_x=sess.run(images)
                feed_dict = {inputs:batch_x, outputs : batch_x}

                _,p_loss,f_loss,reconstruct_imgs=sess.run([opt,pixel_loss,feature_loss,decoded],feed_dict=feed_dict)

                print('step %d |  pixel_loss is %f   | feature_loss is %f  |'%(i,p_loss,f_loss))
                p_loss_list.append(p_loss)
                f_loss_list.append(f_loss)

                if i % 5 ==0:
                    result_img = np.clip(reconstruct_imgs[0],0,255).astype(np.uint8)
                    imsave('result.jpg',result_img)

            saver.save(sess,self.checkpoint_path)
            coord.request_stop()  
            coord.join(threads)
            
        # write pixel losses and feature losses to files                
        fp = open("loss/p_loss.csv",'w')
        fp.writelines(str(p_loss_list))
        fp.close()
        fp = open("loss/f_loss.csv",'w')
        fp.writelines(str(f_loss_list))
        fp.close()
        
        
parser = ArgumentParser()

parser.add_argument('--target_layer', type=str,
                        dest='target_layer', help='target_layer(such as relu5)',
                        metavar='target_layer', required=True)
parser.add_argument('--pretrained_path',type=str,
                        dest='pretrained_path',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
parser.add_argument('--max_iterator',type=int,
                        dest='max_iterator',help='the max iterator',
                        metavar='MAX',required = True)
parser.add_argument('--checkpoint_path',type=str,
                        dest='checkpoint_path',help='checkpoint path',
                        metavar='CheckPoint',required = True)
parser.add_argument('--tfrecord_path',type=str,
                        dest='tfrecord_path',help='tfrecord path',
                        metavar='Tfrecord',required = True)
parser.add_argument('--batch_size',type=int,
                        dest='batch_size',help='batch_size',
                        metavar='Batch_size',required = True)
    


def main():

    opts = parser.parse_args()
    
    model = Train_Model(target_layer = opts.target_layer,
                      pretrained_path = opts.pretrained_path,
                      max_iterator = opts.max_iterator,
                      checkpoint_path = opts.checkpoint_path,
                      tfrecord_path = opts.tfrecord_path,
                      batch_size = opts.batch_size)
    
    model.train()
    
if __name__=='__main__' :
    main()
    
    
