import os
import pandas as pd
import numpy as np


def get_images_in_directory(directory):

    # define image array
    images = []

    # build image array
    for filename in os.listdir(directory + "\\"): # for images in the given directory
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')): # if it has an image file extension
            images.append(os.path.join(directory, filename)) # append the image array with the full path of the image
    return images # return image array

def buildBindingBoxAr(imAr, DF):
    # defining arrays for
    ux = [] # upper left x corrdinate, from top of image
    uy = [] # upper left y coordinate
    lx = [] # lower right x coordinate
    ly = [] # lower right y coordinate
    
    # from 0 to (the number of images)-1
    for i in range(len(imAr)):

        # split the ith image in the image array using delimiter before image name so it matche sthe dataframe
        imgs = imAr[i].split('\\')
        imgs = imgs[-1] # get just the image name

        # assign cooridates from DF to its relevant array
        ux[i] = DF.loc[imgs[i]]['ux']
        uy[i] = DF.loc[imgs[i]]['uy']
        lx[i] = DF.loc[imgs[i]]['lx']
        ly[i] = DF.loc[imgs[i]]['ly']
    
    # convert arrays to numpy arrays for PyTorch
    ux = np.array(ux)
    uy = np.array(uy)
    lx = np.array(lx)
    ly = np.array(ly)

    # transpose arrays for the sake of bounding box assignment
    ux = np.transpose(ux)
    uy = np.transpose(uy)
    lx = np.transpose(lx)
    ly = np.transpose(ly) 
    
    # define array to return and return it
    bbAr = [ux, uy, lx, ly]

    return bbAr