import tensorflow as tf
import configuration as config
from fold import layers
from tensorflow.contrib.layers.python.layers import initializers
slim = tf.contrib.slim
import logging


def unet2D_valid(images, training, nlabels):
    
    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')     #92 with 212x212

    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    logging.info('conv2_2')
    logging.info(conv2_2.shape)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    logging.info('conv5_2')
    logging.info(conv5_2.shape)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape) 
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
    logging.info('conv6_1')
    logging.info(conv6_1.shape)  
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    logging.info('conv6_2')
    logging.info(conv6_2.shape)
                 
    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, padding='VALID')

    return pred

#same
def unet2D_same(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)
    
    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    concat4 = layers.crop_and_concat_layer([conv4_2, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    logging.info('pred')
    logging.info(pred.shape)
    
    return pred


#same reduced
def unet2D_same_mod(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=48, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=48, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=96, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=96, training=training)
    
    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=192, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=192, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=384, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=384, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=768, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=768, training=training)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = layers.crop_and_concat_layer([conv4_2, upconv4], axis=3)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=384, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=384, training=training)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat3 = layers.crop_and_concat_layer([conv3_2, upconv3], axis=3)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=192, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=192, training=training)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = layers.crop_and_concat_layer([conv2_2, upconv2], axis=3)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=96, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=96, training=training)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = layers.crop_and_concat_layer([conv1_2, upconv1], axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=48, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=48, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred


def Dunet2D_same_mod(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=32, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=32, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=64, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=64, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)
    
    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=128, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=128, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=256, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=256, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=512, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=512, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    skip4 = layers.res_net_block(conv4_2, name= 'skip4', num_filters=256, training=training)
    logging.info('skip4')
    logging.info(skip4.shape)
    concat4 = layers.crop_and_concat_layer([skip4, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=256, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=256, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    skip3_1 = layers.res_net_block(conv3_2, name= 'skip3_1', num_filters=128, training=training)
    logging.info('skip3_1')
    logging.info(skip3_1.shape)
    skip3_2 = layers.res_net_block(skip3_1, name= 'skip3_2', num_filters=128, training=training)
    logging.info('skip3_2')
    logging.info(skip3_2.shape)
    concat3 = tf.concat([skip3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=128, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=128, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    skip2_1 = layers.res_net_block(conv2_2, name= 'skip2_1', num_filters=64, training=training)
    logging.info('skip2_1')
    logging.info(skip2_1.shape)
    skip2_2 = layers.res_net_block(skip2_1, name= 'skip2_2', num_filters=64, training=training)
    logging.info('skip2_2')
    logging.info(skip2_2.shape)
    skip2_3 = layers.res_net_block(skip2_2, name= 'skip2_3', num_filters=64, training=training)
    logging.info('skip2_3')
    logging.info(skip2_3.shape)
    concat2 = tf.concat([skip2_3, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=64, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=64, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    skip1_1 = layers.res_net_block(conv1_2, name= 'skip1_1', num_filters=32, training=training)
    logging.info('skip1_1')
    logging.info(skip1_1.shape)
    skip1_2 = layers.res_net_block(skip1_1, name= 'skip1_2', num_filters=32, training=training)
    logging.info('skip1_2')
    logging.info(skip1_2.shape)
    skip1_3 = layers.res_net_block(skip1_2, name= 'skip1_3', num_filters=32, training=training)
    logging.info('skip1_3')
    logging.info(skip1_3.shape)
    skip1_4 = layers.res_net_block(skip1_3, name= 'skip1_4', num_filters=32, training=training)
    logging.info('skip1_4')
    logging.info(skip1_4.shape)
    concat1 = tf.concat([skip1_4, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=32, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=32, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    logging.info('pred')
    logging.info(pred.shape)
    
    return pred


def Dunet2D_same_mod2(images, training, nlabels):
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=48, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=48, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=96, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=96, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)

    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=192, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=192, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=384, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=384, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=768, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=768, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    skip4 = layers.res_net_block(conv4_2, name='skip4', num_filters=384, training=training)
    logging.info('skip4')
    logging.info(skip4.shape)
    concat4 = layers.crop_and_concat_layer([skip4, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=384, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=384, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    skip3_1 = layers.res_net_block(conv3_2, name='skip3_1', num_filters=192, training=training)
    logging.info('skip3_1')
    logging.info(skip3_1.shape)
    skip3_2 = layers.res_net_block(skip3_1, name='skip3_2', num_filters=192, training=training)
    logging.info('skip3_2')
    logging.info(skip3_2.shape)
    concat3 = tf.concat([skip3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=192, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=192, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    skip2_1 = layers.res_net_block(conv2_2, name='skip2_1', num_filters=96, training=training)
    logging.info('skip2_1')
    logging.info(skip2_1.shape)
    skip2_2 = layers.res_net_block(skip2_1, name='skip2_2', num_filters=96, training=training)
    logging.info('skip2_2')
    logging.info(skip2_2.shape)
    skip2_3 = layers.res_net_block(skip2_2, name='skip2_3', num_filters=96, training=training)
    logging.info('skip2_3')
    logging.info(skip2_3.shape)
    concat2 = tf.concat([skip2_3, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=96, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=96, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    skip1_1 = layers.res_net_block(conv1_2, name='skip1_1', num_filters=48, training=training)
    logging.info('skip1_1')
    logging.info(skip1_1.shape)
    skip1_2 = layers.res_net_block(skip1_1, name='skip1_2', num_filters=48, training=training)
    logging.info('skip1_2')
    logging.info(skip1_2.shape)
    skip1_3 = layers.res_net_block(skip1_2, name='skip1_3', num_filters=48, training=training)
    logging.info('skip1_3')
    logging.info(skip1_3.shape)
    skip1_4 = layers.res_net_block(skip1_3, name='skip1_4', num_filters=48, training=training)
    logging.info('skip1_4')
    logging.info(skip1_4.shape)
    concat1 = tf.concat([skip1_4, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=48, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=48, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)
    logging.info('pred')
    logging.info(pred.shape)

    return pred


def Dunet2D_same_mod3(images, training, nlabels):
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=48, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=48, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=96, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=96, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)

    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=192, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=192, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=384, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=384, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    upconv3 = layers.deconv2D_layer_bn(conv4_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    skip3_1 = layers.res_net_block(conv3_2, name='skip3_1', num_filters=192, training=training)
    logging.info('skip3_1')
    logging.info(skip3_1.shape)
    concat3 = tf.concat([skip3_1, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=192, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=192, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    skip2_1 = layers.res_net_block(conv2_2, name='skip2_1', num_filters=96, training=training)
    logging.info('skip2_1')
    logging.info(skip2_1.shape)
    skip2_2 = layers.res_net_block(skip2_1, name='skip2_2', num_filters=96, training=training)
    logging.info('skip2_2')
    logging.info(skip2_2.shape)
    concat2 = tf.concat([skip2_2, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=96, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=96, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    skip1_1 = layers.res_net_block(conv1_2, name='skip1_1', num_filters=48, training=training)
    logging.info('skip1_1')
    logging.info(skip1_1.shape)
    skip1_2 = layers.res_net_block(skip1_1, name='skip1_2', num_filters=48, training=training)
    logging.info('skip1_2')
    logging.info(skip1_2.shape)
    skip1_3 = layers.res_net_block(skip1_2, name='skip1_3', num_filters=48, training=training)
    logging.info('skip1_3')
    logging.info(skip1_3.shape)
    concat1 = tf.concat([skip1_3, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=48, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=48, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)
    logging.info('pred')
    logging.info(pred.shape)

    return pred


def unet2D_light(images, training, nlabels):
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)

    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    upconv4 = layers.deconv2D_layer_bn(conv4_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    concat4 = layers.crop_and_concat_layer([conv3_2, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv5_1 = layers.conv2D_layer_bn(concat4, 'conv5_1', num_filters=256, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)

    upconv3 = layers.deconv2D_layer_bn(conv5_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = layers.crop_and_concat_layer([conv2_2, upconv3], axis=3)
    logging.info('concat3')
    logging.info(concat3.shape)

    conv6_1 = layers.conv2D_layer_bn(concat3, 'conv6_1', num_filters=128, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=128, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)

    upconv2 = layers.deconv2D_layer_bn(conv6_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels,
                                       training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = layers.crop_and_concat_layer([conv1_2, upconv2], axis=3)
    logging.info('concat2')
    logging.info(concat2.shape)

    conv7_1 = layers.conv2D_layer_bn(concat2, 'conv7_1', num_filters=64, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=64, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    pred = layers.conv2D_layer_bn(conv7_2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)
    logging.info('pred')
    logging.info(pred.shape)

    return pred
