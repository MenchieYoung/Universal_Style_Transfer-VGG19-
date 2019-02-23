"""
ECBM 4040 
Group YYZA
Group Members: Manqi Yang (my2577), Shiyun Yang (sy2797), Yizhi Zhang (yz3376)
"""

import tensorflow as tf
import numpy as np
from PIL import Image as pil_image
from skimage.io import imsave,imshow,imread
from Encoder import Vgg19
from Decoder import Decoder
from wct import wct_tf
from argparse import ArgumentParser


class style_transfer:
    """
    Style Transfer Class: 
    this class is the main algorithm of our project;
    generates stylized images given content and style images
    """
    
    def __init__(self,content_path,style_path,alpha,pretrained_vgg,output_path) :
        """
        Style Transfer Initialization
        
        Inputs: 
        content_path: path to content images
        style_path: path to style images
        alpha: range=[0, 1]; parameter to control ratio of content vs. stylized images
        pretrained_vgg: path to pre-trained vgg weights
        output_path: path to store result images
        """
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.alpha = alpha
        self.encoder = Vgg19(pretrained_vgg)
        self.decoder = Decoder()  
        # load decoder weights from saved models: give names of models to use
        self.decoder_weights =['models/relu1_1w_2k','models/relu2_1w_2k', 
                               'models/relu3_1w_1w','models/relu4_8w_1w'] #,'models/relu5_8w_2w']
        
    def test(self):
        """
        Test function: generate stylized images using 5 levels (5 encoder-decoder pairs)
        """
        
        content = tf.placeholder('float',[1,304,304,3])
        style = tf.placeholder('float',[1,304,304,3])
        
        # implementation of multi-level stylization: first level is encoder-decoder 5, last level is encoder-decoder 1
        # original paper named this "coarse-to-fine stylization" 
        
#         content_encode_5 = self.encoder.encoder(content,'relu5')
#         style_encode_5 = self.encoder.encoder(style,'relu5')
#         blended_5 = wct_tf(content_encode_5,style_encode_5,self.alpha)
#         stylized_5 ,var_list5= self.decoder.decoder(blended_5,'relu5')
        
        content_encode_4 = self.encoder.encoder(content,'relu4')
        style_encode_4 = self.encoder.encoder(style,'relu4')
        blended_4 = wct_tf(content_encode_4,style_encode_4,self.alpha)
        stylized_4 ,var_list4= self.decoder.decoder(blended_4,'relu4')
        
        content_encode_3 = self.encoder.encoder(stylized_4,'relu3')
        style_encode_3 = self.encoder.encoder(style,'relu3')
        blended_3 = wct_tf(content_encode_3,style_encode_3,self.alpha)
        stylized_3 ,var_list3= self.decoder.decoder(blended_3,'relu3')
        
        content_encode_2 = self.encoder.encoder(stylized_3,'relu2')
        style_encode_2 = self.encoder.encoder(style,'relu2')
        blended_2 = wct_tf(content_encode_2,style_encode_2,self.alpha)
        stylized_2 ,var_list2= self.decoder.decoder(blended_2,'relu2')        
        
        content_encode_1 = self.encoder.encoder(stylized_2,'relu1')
        style_encode_1 = self.encoder.encoder(style,'relu1')
        blended_1 = wct_tf(content_encode_1,style_encode_1,self.alpha)
        stylized_1,var_list1 = self.decoder.decoder(blended_1,'relu1')
        
        saver1 = tf.train.Saver(var_list1)
        saver2 = tf.train.Saver(var_list2)
        saver3 = tf.train.Saver(var_list3)
        saver4 = tf.train.Saver(var_list4)
        #saver5 = tf.train.Saver(var_list5)
        
        with tf.Session()as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            
            # restore decoder weights from models
            saver1.restore(sess,self.decoder_weights[0])
            saver2.restore(sess,self.decoder_weights[1])
            saver3.restore(sess,self.decoder_weights[2])
            saver4.restore(sess,self.decoder_weights[3])
            #saver5.restore(sess,self.decoder_weights[4])
            
            # load content image using PIL package: if image is in greyscale, convert it to RGB; then resize image
            target_size=(304,304,3)
            content_img = pil_image.open(self.content_path)
            if content_img.mode != 'RGB':
                content_img = img.convert('RGB')
            hw_tuple = (target_size[1], target_size[0])
            if content_img.size != hw_tuple:
                content_img = content_img.resize(hw_tuple)
                
            # convert image to numpy array
            content_img = np.asarray(content_img)
            content_img = np.expand_dims(content_img,axis=0)

            # load style image using PIL package: if image is in greyscale, convert it to RGB; then resize image
            target_size=(304,304,3)
            style_img = pil_image.open(self.style_path)
            if style_img.mode != 'RGB':
                style_img = img.convert('RGB')
            hw_tuple = (target_size[1], target_size[0])
            if style_img.size != hw_tuple:
                style_img = style_img.resize(hw_tuple)
            
            # convert image to numpy array
            style_img = np.asarray(style_img)
            style_img = np.expand_dims(style_img,axis=0)    

            feed_dict = {content : content_img , style : style_img}
            
            # generate result images
            result = sess.run(stylized_1,feed_dict= feed_dict)
            result = result[0]
            result = np.clip(result,0,255)/255.

            imsave(self.output_path,result) 
            print('Style Transfer Completed')

            

parser = ArgumentParser()
    
parser.add_argument('--pretrained_vgg',type=str,
                        dest='pretrained_vgg',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
parser.add_argument('--content_path',type=str,
                        dest='content_path',help='the content path',
                        metavar='Content',required = True)
parser.add_argument('--style_path',type=str,
                        dest='style_path',help='style path',
                        metavar='Style',required = True)    
parser.add_argument('--output_path',type=str,
                        dest='output_path',help='output_path',
                        metavar='Output',required = True)    
parser.add_argument('--alpha',type=float,
                        dest='alpha',help='the blended weight',
                        metavar='ALpha',required = True)

def main():
    opts = parser.parse_args()
    
    model = style_transfer(
                     pretrained_vgg = opts.pretrained_vgg,
                     content_path = opts.content_path,
                     style_path = opts.style_path,
                     output_path = opts.output_path,
                     alpha = opts.alpha,
                     )
    model.test()
    
if __name__ == '__main__' :
    main()

        