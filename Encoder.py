"""
ECBM 4040 
Group YYZA
Group Members: Manqi Yang (my2577), Shiyun Yang (sy2797), Yizhi Zhang (yz3376)
"""

import os
import tensorflow as tf
import numpy as np
import inspect
import layer_funcs


class Vgg19:
    """
    VGG19 Encoder class
    """
    
    def __init__(self, vgg19_npy_path=None):
        """
        VGG19 Encoder Initialization: requires pre-trained weights (vgg19.npy)
        
        Input:
        vgg19_npy_path: path to vgg19.npy file
        """
        
        # load parameters from vgg19.npy file
        self.encoder_param = np.load(vgg19_npy_path, encoding='latin1').item()
        # import convolutional layer and maxpool layer from layer_funcs
        self.conv_layer = layer_funcs.conv_layer
        self.max_pool = layer_funcs.max_pool
        print("VGG19 pre-trained weights loaded")
        
    def encoder(self,inputs,target_layer):
        """
        VGG19 Encoder: 
        
        Inputs: 
        inputs: image files to be encoded to feature maps
        target_layer: indicate which encoder to use (decoder1 -> relu1, ..., decoder5 -> relu5)
        
        Output: 
        encode: output encoded result
        """
        
        # dictinary to look up target layer number
        layer_num =dict(zip(['relu1','relu2','relu3','relu4','relu5'],range(1,6)))[target_layer]
        encode = inputs
        
        # encoder arguments: layers and sizes for each encoder
        encoder_arg={
                '1':[('conv1_1',64),
                     ('conv1_2',64),
                     ('pool1',64)],
                '2':[('conv2_1',128),
                     ('conv2_2',128),
                     ('pool2',128)],
                '3':[('conv3_1',256),
                     ('conv3_2',256),
                     ('conv3_3',256),
                     ('conv3_4',256),
                     ('pool3',256)],                    
                '4':[('conv4_1',512),
                     ('conv4_2',512),
                     ('conv4_3',512),
                     ('conv4_4',512),
                     ('pool4',512)],
                '5':[('conv5_1',512),
                     ('conv5_2',512),
                     ('conv5_3',512),
                     ('conv5_4',512),]}  
                
        # process inputs through each layer in encoder
        for d in range(1,layer_num+1):
            for layer in encoder_arg[str(d)]:                
                if 'conv' in layer[0] :
                    encode = self.conv_layer(encode,self.encoder_param,layer[0])
                if 'pool' in layer[0] and d <layer_num :
                    encode = self.max_pool(encode,layer[0])               
        return encode
    

            
   