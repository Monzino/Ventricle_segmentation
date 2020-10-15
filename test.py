import os
import glob
import numpy as np
import logging
import cv2

import argparse
import metrics
import time
from importlib.machinery import SourceFileLoader
import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt

import configuration as config
import model as model
import utils
import read_data
import image_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def score_data(input_folder, output_folder, model_path, config, do_postprocessing=False, gt_exists=True):

    nx, ny = config.image_size[:2]
    batch_size = 1
    num_channels = config.nlabels

    image_tensor_shape = [batch_size] + list(config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    # According to the experiment config, pick a model and predict the output
    mask_pl, softmax_pl = model.predict(images_pl, config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])

        total_time = 0
        total_volumes = 0
        scale_vector = [config.pixel_size[0] / target_resolution[0], config.pixel_size[1] / target_resolution[1]]

        path_img = os.path.join(input_folder, 'img')
        if gt_exists:
            path_mask = os.path.join(input_folder, 'mask')
        
        for folder in os.listdir(path_img):
            
            logging.info(' ----- Doing image: -------------------------')
            logging.info('Doing: %s' % folder)
            logging.info(' --------------------------------------------')
            folder_path = os.path.join(path_img, folder)   #ciclo su cartelle paz
            
            utils.makefolder(os.path.join(path_pred, folder))
            
            if os.path.isdir(folder_path):
                
                for phase in os.listdir(folder_path):    #ciclo su cartelle ED ES
                    
                    save_path = os.path.join(path_pred, folder, phase)
                    utils.makefolder(save_path)
                    
                    predictions = []
                    mask_arr = []
                    img_arr = []
                    masks = []
                    imgs = []
                    path = os.path.join(folder_path, phase)
                    for file in os.listdir(path):
                        img = plt.imread(os.path.join(path,file))
                        if config.standardize:
                            img = image_utils.standardize_image(img)
                        if config.normalize:
                            img = cv2.normalize(img, dst=None, alpha=config.min, beta=config.max, norm_type=cv2.NORM_MINMAX)
                        img_arr.append(img)
                    if  gt_exists:
                        for file in os.listdir(os.path.join(path_mask,folder,phase)):
                            mask_arr.append(plt.imread(os.path.join(path_mask,folder,phase,file)))
                    
                    img_arr = np.transpose(np.asarray(img_arr),(1,2,0))      # x,y,N
                    if  gt_exists:
                        mask_arr = np.transpose(np.asarray(mask_arr),(1,2,0))
                    
                    start_time = time.time()
                    
                    if config.data_mode == '2D':
                        
                        for zz in range(img_arr.shape[2]):
                            
                            slice_img = np.squeeze(img_arr[:,:,zz])
                            slice_rescaled = transform.rescale(slice_img,
                                                               scale_vector,
                                                               order=1,
                                                               preserve_range=True,
                                                               multichannel=False,
                                                               anti_aliasing=True,
                                                               mode='constant')
                                
                            slice_mask = np.squeeze(mask_arr[:, :, zz])
                            slice_cropped = read_data.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                            slice_cropped = np.float32(slice_cropped)
                            x = image_utils.reshape_2Dimage_to_tensor(slice_cropped)
                            imgs.append(np.squeeze(x))
                            if gt_exists:
                                mask_rescaled = transform.rescale(slice_mask,
                                                                  scale_vector,
                                                                  order=0,
                                                                  preserve_range=True,
                                                                  multichannel=False,
                                                                  anti_aliasing=True,
                                                                  mode='constant')

                                mask_cropped = read_data.crop_or_pad_slice_to_size(mask_rescaled, nx, ny)
                                mask_cropped = np.asarray(mask_cropped, dtype=np.uint8)
                                y = image_utils.reshape_2Dimage_to_tensor(mask_cropped)
                                masks.append(np.squeeze(y))
                                
                            # GET PREDICTION
                            feed_dict = {
                            images_pl: x,
                            }
                                
                            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                            
                            prediction_cropped = np.squeeze(logits_out[0,...])

                            # ASSEMBLE BACK THE SLICES
                            slice_predictions = np.zeros((nx,ny,num_channels))
                            slice_predictions = prediction_cropped
                            # RESCALING ON THE LOGITS
                            if gt_exists:
                                prediction = transform.resize(slice_predictions,
                                                              (nx, ny, num_channels),
                                                              order=1,
                                                              preserve_range=True,
                                                              anti_aliasing=True,
                                                              mode='constant')
                            else:
                                prediction = transform.rescale(slice_predictions,
                                                               (1.0/scale_vector[0], 1.0/scale_vector[1], 1),
                                                               order=1,
                                                               preserve_range=True,
                                                               multichannel=False,
                                                               anti_aliasing=True,
                                                               mode='constant')

                            prediction = np.uint8(np.argmax(prediction, axis=-1))
                            
                            predictions.append(prediction)
                            
       
                        predictions = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
                        masks = np.transpose(np.asarray(masks, dtype=np.uint8), (1,2,0))
                        imgs = np.transpose(np.asarray(imgs, dtype=np.float32), (1,2,0))                   

                            
                    # This is the same for 2D and 3D
                    if do_postprocessing:
                        predictions = image_utils.keep_largest_connected_components(predictions)

                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    total_volumes += 1

                    logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                    
                    # Save predicted mask
                    for ii in range(predictions.shape[2]):
                        image_file_name = os.path.join('paz', str(ii).zfill(3) + '.png')
                        cv2.imwrite(os.path.join(save_path , image_file_name), np.squeeze(predictions[:,:,ii]))
                                        
                    if gt_exists:

               
'''
                        for zz in range(difference_mask.shape[2]):
                            plt.imshow(img_arrs[:,:,zz])
                            plt.gray()
                            plt.axis('off')
                            plt.show()
                            plt.imshow(mask_arrs[:,:,zz])
                            plt.gray()
                            plt.axis('off')
                            plt.show()
                            plt.imshow(prediction_arr[:,:,zz])
                            plt.gray()
                            plt.axis('off')
                            plt.show()
                            print('...')
'''

        logging.info('Average time per volume: %f' % (total_time/total_volumes))

    return init_iteration


if __name__ == '__main__':

    base_path = config.project_root
    logging.info(base_path)
    model_path = config.weights_root
    logging.info(model_path)

    logging.warning('EVALUATING ON TEST SET')
    input_path = config.test_data_root
    output_path = os.path.join(model_path, 'predictions')

    path_pred = os.path.join(output_path, 'prediction')
    utils.makefolder(path_pred)
    path_eval = os.path.join(output_path, 'eval')
        
    gt_exists = config.gt_exists      #True if it exists the ground_truth images, otherwise set False.
                                      #if True it will be defined evalutation (eval)
   

    init_iteration = score_data(input_path,
                                output_path,
                                model_path,
                                config=config,
                                do_postprocessing=True,
                                gt_exists)


    if gt_exists:
        logging.info('Evaluation of the test images')
        path_gt = os.path.join(config.test_data_root , 'mask')
        metrics.main(path_gt, path_pred, path_eval)
