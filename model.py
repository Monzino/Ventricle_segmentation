import tensorflow as tf
from fold import losses
import configuration as config
slim = tf.contrib.slim
import model_structure
import logging
import tensorflow.examples.tutorials.mnist


def inference(images, config, training):
    
    return config.model_handle(images, training, nlabels=config.nlabels)


def loss(logits, labels, nlabels, loss_type, weight_decay=0.0):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'weighted_crossentropy'/'crossentropy'/'dice'/'dice_onlyfg'/'crossentropy_and_dice'
    :param weight_decay: The weight for the L2 regularisation of the network paramters
    :return: The total loss including weight decay, the loss without weight decay, only the weight decay 
    '''

    labels = tf.one_hot(labels, depth=nlabels)

    with tf.compat.v1.variable_scope('weights_norm'):

        weights_norm = tf.reduce_sum(
            input_tensor = weight_decay*tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.compat.v1.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

    if loss_type == 'weighted_crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(logits, labels,
                                                                          class_weights=[0.076, 0.308, 0.308, 0.308])
    elif loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels, only_foreground=False)
    elif loss_type == 'dice_onlyfg':
        segmentation_loss = losses.dice_loss(logits, labels, only_foreground=True)
    elif loss_type == 'crossentropy_and_dice':
        segmentation_loss = config.alfa*losses.pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights=[0.076, 0.308, 0.308, 0.308]) + config.beta*losses.dice_loss(logits, labels, only_foreground=True)
    else:
        raise ValueError('Unknown loss: %s' % loss_type)


    total_loss = tf.add(segmentation_loss, weights_norm)

    return total_loss, segmentation_loss, weights_norm


def predict(images, config):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''
    if (config.experiment_name == 'unet2D_valid' or config.experiment_name == 'unet2D_same' or config.experiment_name == 'unet2D_same_mod' or config.experiment_name == 'unet2D_light' or config.experiment_name == 'Dunet2D_same_mod' or config.experiment_name == 'Dunet2D_same_mod2' or config.experiment_name == 'Dunet2D_same_mod3'):
        logits = inference(images, config, training=tf.constant(False, dtype=tf.bool))
    else:
        logging.warning('invalid experiment_name!') 
    softmax = tf.nn.softmax(logits)
    mask = tf.math.argmax(softmax, axis=-1)

    return mask, softmax


def training_step(loss, optimizer_handle, lr, **kwargs):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param lr: Learning rate
    :param momentum: Optionally, you can also pass a momentum term to the optimiser. 
    :return: The training operation
    '''
    
    if config.exponential_decay:
        global_step = tf.train.get_or_create_global_step()
        num_epochs_before_decay = int(config.max_epochs/3)
        num_steps_per_epoch = 1500/config.batch_size 
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
        lr = tf.train.exponential_decay(learning_rate = config.learning_rate,
                                        global_step = global_step,
                                        decay_steps = decay_steps,
                                        decay_rate = 1e-1,
                                        staircase = True)
    
    if 'momentum' in kwargs:
        momentum = kwargs.get('momentum')
        optimizer = optimizer_handle(learning_rate=lr, momentum=momentum)
    else:
        optimizer = optimizer_handle(learning_rate=lr)

    # The with statement is needed to make sure the tf contrib version of batch norm properly performs its updates
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    return train_op


def evaluation(logits, labels, images, nlabels, loss_type):
    '''
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the 
    current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    '''

    mask = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)  # was 3
    mask_gt = labels

    tf.compat.v1.summary.image('example_gt', prepare_tensor_for_summary(mask_gt, mode='mask', nlabels=nlabels))
    tf.compat.v1.summary.image('example_pred', prepare_tensor_for_summary(mask, mode='mask', nlabels=nlabels))
    tf.compat.v1.summary.image('example_zimg', prepare_tensor_for_summary(images, mode='image'))

    total_loss, nowd_loss, weights_norm = loss(logits, labels, nlabels=nlabels, loss_type=loss_type)

    cdice_structures = losses.per_structure_dice(logits, tf.one_hot(labels, depth=nlabels))
    cdice_foreground = cdice_structures[:,1:]

    cdice = tf.reduce_mean(cdice_foreground)

    return nowd_loss, cdice


def prepare_tensor_for_summary(img, mode, idx=0, nlabels=None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':

        if img.get_shape().ndims == 3:
            V = img[idx, ...]
        elif img.get_shape().ndims == 4:
            V = img[idx, ..., 10]
        elif img.get_shape().ndims == 5:
            V = img[idx, ..., 10, 0]
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':

        if img.get_shape().ndims == 3:
            V = img[idx, ...]
        elif img.get_shape().ndims == 4:
            V = img[idx, ..., 0]
        elif img.get_shape().ndims == 5:
            V = img[idx, ..., 10, 0]
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else:
        raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]

    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
