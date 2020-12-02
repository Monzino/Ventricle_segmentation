# Ventricle_segmentation

## Requirements

    Python 3.5 (tested with Python 3.5.3)
    Tensorflow (tested with tensorflow 1.14)
    The package requirements are given in requirements.txt

## Getting the code

   Clone the repository by typing
    
    git clone https://github.com/Monzino/Ventricle_segmentation

## Installing required Python packages

   Create an environment with Python 3.5. 
   Next, install the required packages listed in the requirements.txt file as:

    pip install -r requirements.txt

   The tensorflow GPU packages can be installed as:

    pip install tensorflow-gpu==1.14
    
   GPU TensorFlow uses CUDA. Please read the CUDA® install guide for Windows 
   
    https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

## Preparing the data for training

   You need to make two folders

   - Images Folder (img)- For all the training images
   - Annotations Folder (mask) - For the corresponding ground truth segmentation images
  
  Inside the Images and Annotation folder, you need to create a folder for each subject, with the corresponding images.
  
  The filenames of the annotation images should be same as the filenames of the images.

  The size of the annotation image for the corresponding image should be same.

## Configuration

   Open configuration.py for the settings, such as the network architecture and the data augmentation. 
   Set all the paths to match your system before run the code. 

## Model

Dense Fully Convolutional Neural Network

![github-small](https://raw.githubusercontent.com/Monzino/Ventricle_segmentation/main/network.png)

Architecture of the proposed segmentation network for a 176x176 pixels input image. Each blue box corresponds to a multi-channel feature map. The number of channels is reported on top of each box.

## Running the code

To train the model simply run 

    train.py
    
To evaluate the model run 

    evaluate_patients.py

## Results

Example of segmentation results of basal slices 

![github-small](https://raw.githubusercontent.com/Monzino/Ventricle_segmentation/main/Picture1.png)

Example of segmentation results of mid-ventricular level 

![github-small](https://raw.githubusercontent.com/Monzino/Ventricle_segmentation/main/Picture2.png)

Example of segmentation results of apical slices 

![github-small](https://raw.githubusercontent.com/Monzino/Ventricle_segmentation/main/Picture3.png)




   

