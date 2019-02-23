"""
ECBM 4040 
Group YYZA
Group Members: Manqi Yang (my2577), Shiyun Yang (sy2797), Yizhi Zhang (yz3376)
"""

import tensorflow as tf

####################################### encoder layers ##################################################

# 1/2 max-pooling layer
def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# convolutional layer
def conv_layer(bottom, encoder_param, name):
    with tf.variable_scope(name):
        filt = get_conv_filter(encoder_param,name)
        filt_size=3
        bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')

        conv_biases = get_bias(encoder_param,name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

# functions used in convolutional layer
def get_conv_filter(encoder_param,name):
    return tf.constant(encoder_param[name][0], name="filter")
def get_bias(encoder_param, name):
    return tf.constant(encoder_param[name][1], name="biases")


########################################## decoder layers ##############################################

# upsampling layer
def upsample(bottom,height):
        height=height
        width=height

        new_height=height*2
        new_width = width*2
        return tf.image.resize_images(bottom, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# output layer
def output_layer(encoder_param, trainable, var_param, bottom, in_channels, out_channels, name,var_list):
    with tf.variable_scope(name):
        filt_size = 9
        filt, conv_biases = get_conv_var(encoder_param, trainable, var_param, filt_size, in_channels, out_channels, name)
        bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
        bias = tf.nn.bias_add(conv, conv_biases)

        var_list.append(filt)
        var_list.append(conv_biases)
        return bias,var_list

# convolutional layer for decoders
def conv_layer_decoder(encoder_param, trainable, var_param, bottom, in_channels, out_channels, name,var_list):
    filt_size = 3
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(encoder_param, trainable, var_param, filt_size, in_channels, out_channels, name)

        bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)  

        var_list.append(filt)
        var_list.append(conv_biases)
        return relu,var_list


# function used in convolutional layer for decoders
def get_conv_var(encoder_param, trainable, var_param, filter_size, in_channels, out_channels, name):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(encoder_param, trainable, var_param, initial_value, name, 0, name + "_filters")

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(encoder_param, trainable, var_param, initial_value, name, 1, name + "_biases")

    return filters, biases


# function used in get_conv_var
def get_var(encoder_param, trainable, var_param, initial_value, name, idx, var_name):
    if encoder_param is not None and name in encoder_param:
        value = encoder_param[name][idx]
        print ('resore %s weight'%(name))
    else:
        value = initial_value

    if trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    var_param[(name, idx)] = var

    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var

