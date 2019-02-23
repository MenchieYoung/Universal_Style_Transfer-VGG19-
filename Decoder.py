"""
ECBM 4040 
Group YYZA
Group Members: Manqi Yang (my2577), Shiyun Yang (sy2797), Yizhi Zhang (yz3376)
"""

import tensorflow as tf
import numpy as np
from functools import reduce
import layer_funcs




class Decoder:
    """
    Reverse VGG Decoder Class
    """
    
    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        """
        Decoder Class Initialization: 
        
        Inputs: 
        vgg19_npy_path: path to vgg19.npy file if using one
        trainable: whether to use trainable version of VGG decoder; default is True
        dropout: dropout probability
        """
        
        # if using pre-trained weights, load vgg19.npy; else None
        if vgg19_npy_path is not None:
            self.encoder_param = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.encoder_param = None

        self.var_param = {}
        self.trainable = trainable
        self.dropout = dropout
        self.upsample = layer_funcs.upsample
        self.conv_layer_decoder = layer_funcs.conv_layer_decoder
        self.output_layer = layer_funcs.output_layer
        
    
    def decoder(self, encode,target_layer) :
        """
        Decoder
        
        Inputs: 
        encode: feature maps to be decoded to image files
        target_layer: indicate which decoder to use (decoder1 -> relu1, ..., decoder5 -> relu5)
        
        Outputs:
        decode: decoded image result
        var_list: list of variables in training
        """
        
        # dictinary to look up target layer number
        layer_num = dict(zip(['relu1','relu2','relu3','relu4','relu5'],range(1,6)))[target_layer]
        var_list=[]
        
        # decoder arguments: layers and sizes for each decoder
        decode_arg={
                '5':[ ('dconv5_1',512,512),
                     ('dconv5_2',512,512),
                     ('dconv5_3',512,512),
                     ('dconv5_4',512,512)],
                
                '4':[  ('upsample',14,28),
                     ('dconv4_1',512,256),
                     ('dconv4_2',256,256),
                     ('dconv4_3',256,256),
                     ('dconv4_4',256,256)],

                '3':[('upsample',28,56),
                     ('dconv3_1',256,128),
                     ('dconv3_2',128,128),
                     ('dconv3_3',128,128),
                     ('dconv3_4',128,128)],

                '2':[('upsample',56,112),
                     ('dconv2_1',128,64),
                     ('dconv2_2',64,64)],
            
                '1':[('upsample',112,224),
                    ('dconv1_1',64,64),
                     ('output',64,3)]} 
        decode = encode
        
        # process inputs through each layer in decoder
        for d in reversed(range(1,layer_num+1)):
            for layer in decode_arg[str(d)]:
                if 'up' in layer[0]:
                    decode = self.upsample(decode,layer[1])
                if 'dconv' in layer[0] :
                    decode ,var_list= self.conv_layer_decoder(self.encoder_param, self.trainable, 
                                                              self.var_param,decode,layer[1],layer[2],layer[0]+'_'+target_layer,var_list)
                if 'out' in layer[0] :
                    decode, var_list = self.output_layer(self.encoder_param, self.trainable, self.var_param, 
                                                         decode,layer[1],layer[2],layer[0]+'_'+target_layer,var_list)
                    
        return decode , var_list

    