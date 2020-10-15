import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
from skimage import util
import cv2
from PIL import Image

import utils
import image_utils
import configuration as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def prepare_data(input_folder, output_file, mode, size, target_resolution):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    hdf5_file = h5py.File(output_file, "w")

    nx, ny = size
    # scale_vector = [config.pixel_size[0] / target_resolution[0], config.pixel_size[1] / target_resolution[1]]
    count = 1
    train_addrs = []
    val_addrs = []
    masktrain_addrs = []
    maskval_addrs = []
    
    # se split_test_train è True allora splitto tra train e validation i pazienti. Quando faccio il test, 
    # split_test_train deve essere False. Split mi dice ogni quanti pazienti vanno in validation. Con 2, il 50% sono divisi.
    # con 5 per esempio uno ogni 5 finisc in validation etc.
    split_test_train = config.split_test_train
    if split_test_train:
        split = config.split
    else:
        split = 99999
    
    path_img = os.path.join(input_folder, 'img')
    path_mask = os.path.join(input_folder, 'mask')
    for folders_img, folders_mask in zip(sorted(os.listdir(path_img)), sorted(os.listdir(path_mask))):
        folder_path_img = os.path.join(path_img, folders_img)
        folder_path_mask = os.path.join(path_mask, folders_mask)
        if count % split == 0:
            #validation
            path = os.path.join(folder_path_img, '*.png')
            for file in sorted(glob.glob(path)):
                val_addrs.append(file)
            path = os.path.join(folder_path_mask, '*.png')
            for file in sorted(glob.glob(path)):
                maskval_addrs.append(file)
        else:
            #training
            path = os.path.join(folder_path_img, '*.png')
            for file in sorted(glob.glob(path)):
                train_addrs.append(file)
            path = os.path.join(folder_path_mask, '*.png')
            for file in sorted(glob.glob(path)):
                masktrain_addrs.append(file)
        
        count = count + 1
    
    train_shape = (len(train_addrs), nx, ny)
    val_shape = (len(val_addrs), nx, ny)
    
    if config.split_test_train:
        if len(train_addrs) != len(masktrain_addrs) or len(val_addrs) != len(maskval_addrs):
            raise AssertionError('Error: Masks and Images have not the same number !!!')
    
    hdf5_file.create_dataset("images_train", train_shape, np.float32)
    hdf5_file.create_dataset("masks_train", train_shape, np.uint8)
    if config.split_test_train:
        hdf5_file.create_dataset("images_val", val_shape, np.float32)
        hdf5_file.create_dataset("masks_val", val_shape, np.uint8)
    
    for i in range(len(train_addrs)):
        addr_img = train_addrs[i]
        addr_mask = masktrain_addrs[i]
        img = cv2.imread(addr_img, 0)   #0 for grayscale
        mask = cv2.imread(addr_mask, 0)
        
        if config.standardize:
            img = image_utils.standardize_image(img)
        if config.normalize:
            img = image_utils.normalize_image(img)
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nx, ny), interpolation=cv2.INTER_NEAREST)

        #img = crop_or_pad_slice_to_size(img, nx, ny)
        #mask = crop_or_pad_slice_to_size(mask, nx, ny)
        hdf5_file["images_train"][i, ...] = img[None]
        hdf5_file["masks_train"][i, ...] = mask[None]
    
    if config.split_test_train:
        for i in range(len(val_addrs)):
            addr_img = val_addrs[i]
            addr_mask = maskval_addrs[i]
            img = cv2.imread(addr_img, 0)
            mask = cv2.imread(addr_mask,0)
            
            if config.standardize:
                img = image_utils.standardize_image(img)
            if config.normalize:
                img = image_utils.normalize_image(img)
            img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (nx, ny), interpolation=cv2.INTER_NEAREST)
            
            #img = crop_or_pad_slice_to_size(img, nx, ny)
            #mask = crop_or_pad_slice_to_size(mask, nx, ny)
            hdf5_file["images_val"][i, ...] = img[None]
            hdf5_file["masks_val"][i, ...] = mask[None]

            
    # After test train loop:
    hdf5_file.close()
    
  
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                force_overwrite=True):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:

        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, mode, size, target_resolution)

    else:

        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    input_folder = config.data_root
    preprocessing_folder = config.preprocessing_folder

    d=load_and_maybe_process_data(input_folder, preprocessing_folder, '2D', config.image_size, config.target_resolution)
