import os.path
from glob import glob
import time
import re
import argparse
import pandas as pd
import h5py
import tensorflow as tf
import shutil
slim = tf.contrib.slim
import glob
import os

import SimpleITK as sitk
from multiprocessing import pool
import pickle
import numpy as np
import logging
import cv2

import utils
import image_utils
import model as model
import read_data
import configuration as config
import augmentation as aug
from background_generator import BackgroundGenerator
import model_structure as model_structure

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(config.log_root, config.experiment_name)

def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % config.experiment_name)

    init_step = 0

    if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0

        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    train_on_all_data = config.train_on_all_data
    
    # Load data
    data = read_data.load_and_maybe_process_data(
        input_folder=config.data_root,
        preprocessing_folder=config.preprocessing_folder,
        mode=config.data_mode,
        size=config.image_size,
        target_resolution=config.target_resolution,
        force_overwrite=False
    )
    
    # the following are HDF5 datasets, not numpy arrays
    images_train = data['images_train']
    labels_train = data['masks_train']

    if not train_on_all_data:
        images_val = data['images_val']
        labels_val = data['masks_val']
        

    if config.use_data_fraction:
        num_images = images_train.shape[0]
        new_last_index = int(float(num_images)*config.use_data_fraction)

        logging.warning('USING ONLY FRACTION OF DATA!')
        logging.warning(' - Number of imgs orig: %d, Number of imgs new: %d' % (num_images, new_last_index))
        images_train = images_train[0:new_last_index,...]
        labels_train = labels_train[0:new_last_index,...]

    logging.info('Data summary:')
    logging.info(' - Images:')
    logging.info(images_train.shape)
    logging.info(images_train.dtype)
    logging.info(' - Labels:')
    logging.info(labels_train.shape)
    logging.info(labels_train.dtype)
    
   
    
 #   if config.prob:   #if prob is not 0
 #       logging.info('Before data_augmentation the number of training images is:')
 #       logging.info(images_train.shape[0])
 #       #augmentation
 #       image_aug, label_aug = aug.augmentation_function(images_train,labels_train)
    
        #num_aug = image_aug.shape[0]
        # id images augmented will be b'0.0'
        #id_aug = np.zeros([num_aug,]).astype('|S9')
        #concatenate
        #id_train = np.concatenate((id__train,id_aug))
 #       images_train = np.concatenate((images_train,image_aug))
 #       labels_train = np.concatenate((labels_train,label_aug))
    
 #       logging.info('After data_augmentation the number of training images is:')
 #       logging.info(images_train.shape[0])
 #   else:
 #       logging.info('No data_augmentation. Number of training images is:')
 #       logging.info(images_train.shape[0])


    # Tell TensorFlow that the model will be built into the default Graph.

    with tf.Graph().as_default():

        # Generate placeholders for the images and labels.

        image_tensor_shape = [config.batch_size] + list(config.image_size) + [1]
        mask_tensor_shape = [config.batch_size] + list(config.image_size)

        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')

        learning_rate_pl = tf.placeholder(tf.float32, shape=[])
        training_pl = tf.placeholder(tf.bool, shape=[])

        tf.summary.scalar('learning_rate', learning_rate_pl)

        # Build a Graph that computes predictions from the inference model.
        if (config.experiment_name == 'unet2D_valid' or config.experiment_name == 'unet2D_same' or config.experiment_name == 'unet2D_same_mod' or config.experiment_name == 'unet2D_light' or config.experiment_name == 'Dunet2D_same_mod' or config.experiment_name == 'Dunet2D_same_mod2' or config.experiment_name == 'Dunet2D_same_mod3'):
            logits = model.inference(images_pl, config, training=training_pl)
        elif config.experiment_name == 'ENet':
            with slim.arg_scope(model_structure.ENet_arg_scope(weight_decay=2e-4)):
                logits = model_structure.ENet(images_pl,
                                              num_classes=config.nlabels,
                                              batch_size=config.batch_size,
                                              is_training=True,
                                              reuse=None,
                                              num_initial_blocks=1,
                                              stage_two_repeat=2,
                                              skip_connections=config.skip_connections)
        else:
            logging.warning('invalid experiment_name!')    
        
        logging.info('images_pl shape')
        logging.info(images_pl.shape)
        logging.info('labels_pl shape')
        logging.info(labels_pl.shape)
        logging.info('logits shape:')
        logging.info(logits.shape)
        # Add to the Graph the Ops for loss calculation.
        [loss, _, weights_norm] = model.loss(logits,
                                             labels_pl,
                                             nlabels=config.nlabels,
                                             loss_type=config.loss_type,
                                             weight_decay=config.weight_decay)  # second output is unregularised loss
        
        # record how Total loss and weight decay change over time
        tf.summary.scalar('loss', loss)  
        tf.summary.scalar('weights_norm_term', weights_norm)

        # Add to the Graph the Ops that calculate and apply gradients.
        if config.momentum is not None:
            train_op = model.training_step(loss, config.optimizer_handle, learning_rate_pl, momentum=config.momentum)
        else:
            train_op = model.training_step(loss, config.optimizer_handle, learning_rate_pl)

        # Add the Op to compare the logits to the labels during evaluation.
        # loss and dice on a minibatch
        eval_loss = model.evaluation(logits,
                                     labels_pl,
                                     images_pl,
                                     nlabels=config.nlabels,
                                     loss_type=config.loss_type)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.

        if train_on_all_data:
            max_to_keep = None
        else:
            max_to_keep = 5

        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver_best_dice = tf.train.Saver()
        saver_best_xent = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        configP = tf.ConfigProto()
        configP.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        configP.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
        sess = tf.Session(config=configP)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # with tf.name_scope('monitoring'):

        val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error')
        val_error_summary = tf.summary.scalar('validation_loss', val_error_)

        val_dice_ = tf.placeholder(tf.float32, shape=[], name='val_dice')
        val_dice_summary = tf.summary.scalar('validation_dice', val_dice_)

        val_summary = tf.summary.merge([val_error_summary, val_dice_summary])

        train_error_ = tf.placeholder(tf.float32, shape=[], name='train_error')
        train_error_summary = tf.summary.scalar('training_loss', train_error_)

        train_dice_ = tf.placeholder(tf.float32, shape=[], name='train_dice')
        train_dice_summary = tf.summary.scalar('training_dice', train_dice_)

        train_summary = tf.summary.merge([train_error_summary, train_dice_summary])

        # Run the Op to initialize the variables.
        sess.run(init)

        if continue_run:
            # Restore session
            saver.restore(sess, init_checkpoint_path)

        step = init_step
        curr_lr = config.learning_rate

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf
        best_dice = 0

        for epoch in range(config.max_epochs):

            logging.info('EPOCH %d' % epoch)


            for batch in iterate_minibatches(images_train,
                                             labels_train,
                                             batch_size=config.batch_size,
                                             augment_batch=config.augment_batch):

                if config.warmup_training:
                    if step < 50:
                        curr_lr = config.learning_rate / 10.0
                    elif step == 50:
                        curr_lr = config.learning_rate

                start_time = time.time()

                # batch = bgn_train.retrieve()
                x, y = batch

                # TEMPORARY HACK (to avoid incomplete batches)
                if y.shape[0] < config.batch_size:
                    step += 1
                    continue

                feed_dict = {
                    images_pl: x,
                    labels_pl: y,
                    learning_rate_pl: curr_lr,
                    training_pl: True
                }


                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 20 == 0:
                    # Print status to stdout.
                    logging.info('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.

                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                
                if (step + 1) % config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_train,
                                                       labels_train,
                                                       config.batch_size)

                    train_summary_msg = sess.run(train_summary, feed_dict={train_error_: train_loss,
                                                                           train_dice_: train_dice}
                                                 )
                    summary_writer.add_summary(train_summary_msg, step)

                    loss_history.append(train_loss)
                    if len(loss_history) > 5:
                        loss_history.pop(0)
                        loss_gradient = (loss_history[-5] - loss_history[-1]) / 2

                    logging.info('loss gradient is currently %f' % loss_gradient)

                    if config.schedule_lr and loss_gradient < config.schedule_gradient_threshold:
                        logging.warning('Reducing learning rate!')
                        curr_lr /= 10.0
                        logging.info('Learning rate changed to: %f' % curr_lr)

                        # reset loss history to give the optimisation some time to start decreasing again
                        loss_gradient = np.inf
                        loss_history = []

                    if train_loss <= last_train:  # best_train:
                        no_improvement_counter = 0
                        logging.info('Decrease in training error!')
                    else:
                        no_improvement_counter = no_improvement_counter+1
                        logging.info('No improvment in training error for %d steps' % no_improvement_counter)

                    last_train = train_loss

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % config.val_eval_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.

                    if not train_on_all_data:

                        # Evaluate against the validation set.
                        logging.info('Validation Data Eval:')
                        [val_loss, val_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_val,
                                                       labels_val,
                                                       config.batch_size)

                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss, val_dice_: val_dice}
                        )
                        summary_writer.add_summary(val_summary_msg, step)

                        if val_dice > best_dice:
                            best_dice = val_dice
                            best_file = os.path.join(log_dir, 'model_best_dice.ckpt')
                            filelist = glob.glob(os.path.join(log_dir, 'model_best_dice*'))
                            for file in filelist:
                                os.remove(file)
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - %f -  Saving model_best_dice.ckpt' % val_dice)

                        if val_loss < best_val:
                            best_val = val_loss
                            best_file = os.path.join(log_dir, 'model_best_xent.ckpt')
                            filelist = glob.glob(os.path.join(log_dir, 'model_best_xent*'))
                            for file in filelist:
                                os.remove(file)
                            saver_best_xent.save(sess, best_file, global_step=step)
                            logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)

                step += 1
                
            # end epoch
            if (epoch + 1) % config.epoch_freq == 0:
                curr_lr = curr_lr *0.98
            logging.info('Learning rate: %f' % curr_lr)
        sess.close()
    data.close()


def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h5py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in BackgroundGenerator(iterate_minibatches(images, labels, batch_size=batch_size, augment_batch=False)):  # No aug in evaluation
    # you can wrap the iterate_minibatches function in the BackgroundGenerator class for speed improvements
    # but at the risk of not catching exceptions

        x, y = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = { images_placeholder: x,
                      labels_placeholder: y,
                      training_time_placeholder: False}

        closs, cdice = sess.run(eval_loss, feed_dict=feed_dict)
        loss_ii += closs
        dice_ii += cdice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice


def iterate_minibatches(images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :return: mini batches
    '''

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]
        #Xid = id_img[batch_indices]

        image_tensor_shape = [X.shape[0]] + list(config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)
        
        if augment_batch:
            X, y = aug.augmentation_function(X, y)

            
        yield X, y


def main():

    continue_run = True
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description="Train a neural network.")
    # parser.add_argument("CONFIG_PATH", type=str, help="Path to config file (assuming you are in the working directory)")
    # args = parser.parse_args() 
    main()
