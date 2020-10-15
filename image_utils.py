import numpy as np
from skimage import measure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:
    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)


    def resize_image(im, size, interp=cv2.INTER_LINEAR):

        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        return im_resized


def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)

def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def normalize_image(image):
    '''
    make image normalize between 0 and 1
    '''
    img_o = np.float32(image.copy())
    #img_o = 2*((img_o-img_o.min())/(img_o.max()-img_o.min()))-1
    img_o = (img_o-img_o.min())/(img_o.max()-img_o.min())
    return img_o

def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,:,:,:]
        mc = Xc.mean()
        sc = Xc.std()

        Xc_white = np.divide((Xc - mc), sc)

        X_white[ii,:,:,:] = Xc_white

    return X_white.astype(np.float32)


def reshape_2Dimage_to_tensor(image):
    return np.reshape(image, (1,image.shape[0], image.shape[1],1))


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    for zz in range(mask.shape[2]):
        img = mask[:,:,zz]
        out_img = np.zeros(img.shape, dtype=np.uint8)

        for struc_id in [1, 2, 3]:

            binary_img = img == struc_id
            blobs = measure.label(binary_img, connectivity=1)
            props = measure.regionprops(blobs)
            if not props:
                continue
            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label
            out_img[blobs == largest_blob_label] = struc_id
        mask[:,:,zz] = out_img
    return mask
