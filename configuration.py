import model_structure
import tensorflow as tf
import os
import socket
import logging

#experiment_name = 'unet2D_valid'
#experiment_name = 'unet2D_same'
experiment_name = 'unet2D_same_mod'
#experiment_name = 'unet2D_light'
#experiment_name = 'Dunet2D_same_mod'
#experiment_name = 'Dunet2D_same_mod2'
#experiment_name = 'Dunet2D_same_mod3'
#experiment_name = 'ENet'

# Model settings Unet2D
weight_init = 'he_normal'    # xavier_uniform/ xavier_normal/ he_normal /he_uniform /caffe_uniform/ simple/ bilinear
#model_handle = model_structure.unet2D_valid
#model_handle = model_structure.unet2D_same
model_handle = model_structure.unet2D_same_mod
#model_handle = model_structure.unet2D_light
#model_handle = model_structure.Dunet2D_same_mod
#model_handle = model_structure.Dunet2D_same_mod2
#model_handle = model_structure.Dunet2D_same_mod3

# Data settings
data_mode = '2D' 
image_size = (176, 176)   #(212,212)
target_resolution = (1, 1)
pixel_size = (1,1) 
nlabels = 4
split_test_train = True   #divide patients in train and test. If true define split
split = 5                 #  2: 50% train and 50% validation,    5: 80% training, 20% validation
train_on_all_data = False 
gt_exists = True    #True if it exists the ground_trth images, otherwise False.
z_dim = 8

# Training settings
batch_size = 4      #4 
learning_rate = 0.01   #unet: 0.01    enet: 0.0005
exponential_decay = False     #True Enet
optimizer_handle = tf.compat.v1.train.AdamOptimizer     #(beta1 = 0.9, beta2 = 0.999, epsilon=1e-08)
schedule_lr = False    #decrease 10 times the LR when loss gradient lower than threshold
warmup_training = True
weight_decay = 0  # enet:2e-4    #unet: 0.00000
momentum = None
# loss can be 'weighted_crossentropy'/'crossentropy'/'dice'/'dice_onlyfg'/'crossentropy_and_dice (alfa,beta)'
loss_type = 'weighted_crossentropy'
alfa = 1     #1      
beta = 0.2   #1      
augment_batch = True

# Augmentation settings
do_rotation_range = True   #random rotation in range "rg" (min,max)
rg = (-15,15)     
do_rotation_90 = False      #rotation 90°
do_rotation_180 = False     #rotation 180°
do_rotation_270 = False     #rotation 270°
do_rotation_reshape = False #rotation of a specific 'angle' with reshape
do_rotation = False         #rotation of a specific 'angle'
angle = 45
crop = False                #crops/cuts away pixels at the sides of the image
do_fliplr = False           #Flip array in the left/right direction
do_flipud = False           #Flip array in the up/down direction.
RandomContrast= False       #Random change contrast of an image
min_factor = 1.0
max_factor = 1.0
blurr = True               #Blurring the image with gaussian filter with random 'sigma'
SaltAndPepper = False
density = 0.05              #Noise density for salt and pepper noise, specified as a numeric scalar.
Multiply = False            #Multiply all pixels in an image with a specific value (m)
m = 1
gamma = True
ElasticTransformation = False #Moving pixels locally around using displacement fields.
alpha = (0.0, 70.0)         #alpha and sigma can be a number or tuple (a, b)
sigma = 5.0                 #If tuple a random value from range ``a <= x <= b`` will be used
Pad = False                 #Pad image, i.e. adds columns/rows to them
offset2 = (10,30)           #number of pixels to crop away on each side of the image (a,b)
                            #each side will be cropped by a random amount in the range `a <= x <= b`
  
prob = 1                    #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (need to mount MyDrive before)
data_root = '/content/drive/My Drive/Pazienti/train2.1'      
test_data_root = '/content/drive/My Drive/Pazieni/test2.1'
preprocessing_folder = '/content/drive/My Drive/preproc_data'     
project_root = '/content/drive/My Drive'                       
log_root = os.path.join(project_root, 'acdc_logdir')
weights_root = os.path.join(log_root, experiment_name)

# Pre-process settings
standardize = False
normalize = True

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_epochs = 3000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 150
epoch_freq = 5     
