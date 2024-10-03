#-----------------------------------------------------
#Import external libraries
import cv2
import numpy
import os
import math

#-----------------------------------------------------


#Function to read image from file
def read_single_image(filename=None, forced_size=None, scaled=True, info=False):
    
    #Error check
    if not isinstance(filename, str):
        err_msg = "Filename is not in string format"
        raise ValueError(err_msg)
    
    #Read image
    image_array = cv2.imread(filename)

    #Scaling
    if(forced_size!=None):
        image_array = cv2.resize(image_array,(forced_size[1],forced_size[0]))

    #Scaling
    if(scaled==True):
        image_array = image_array/255

    #Display
    if(info==True):
        if(image_array.shape[-1] == 1):
            print("Your current image is grayscale and has the size: (" + str(image_array.shape[0]) + "," + str(image_array.shape[1]) + ")")
        elif(image_array.shape[-1] == 3):
            print("Your current image is RGB and has the size: (" + str(image_array.shape[0]) + "," + str(image_array.shape[1]) + ")")
        elif(image_array.shape[-1] > 3):
            print("Your current image is hyperspectral and has the size: (" + str(image_array.shape[0]) + "," + str(image_array.shape[1]) + ")")
        else:
            print("Your current image is customized and has the size: (" + str(image_array.shape[0]) + "," + str(image_array.shape[1]) + ")")

    #Check size
    if(forced_size==None):

        #Return image array
        return image_array
    
    else:

        if(math.log2(image_array.shape[0]).is_integer())&(math.log2(image_array.shape[1]).is_integer()):
            
            #Return image array
            return image_array
        
        else:
            
            #Error check
            err_msg = "Image size is not a power of 2. Please resize!"
            raise ValueError(err_msg)


#Function to read label from file
def read_single_label(filename=None, forced_size=None, scaled=True, info=False):
    
    #Error check
    if not isinstance(filename, str):
        err_msg = "Filename is not in string format"
        raise ValueError(err_msg)
    
    #Read image
    image_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    #Scaling
    if(forced_size!=None):
        image_array = cv2.resize(image_array,(forced_size[1],forced_size[0]))

    #Create binary image
    if(scaled==True):
        image_array = image_array/255
        image_array[image_array < 0.5]  = 0
        image_array[image_array >= 0.5] = 1

    #Display
    if(info==True):
        if(image_array.ndim < 3):
            print("Your current label has the size: (" + str(image_array.shape[0]) + "," + str(image_array.shape[1]) + ")")
        else:
            print("Warning! It is not a labeled image. Better to change it!")

    #Check size
    if(forced_size==None):

        #Return image array
        return image_array
    
    else:

        if(math.log2(image_array.shape[0]).is_integer())&(math.log2(image_array.shape[1]).is_integer()):
            
            #Return image array
            return image_array
        
        else:
            
            #Error check
            err_msg = "Image size is not a power of 2. Please resize!"
            raise ValueError(err_msg)


#Function to read image from folder
def read_images_from_folder(folder_dir=None, forced_size=None, scaled=True):
    
    #Display
    if(forced_size!=None):
        print("Warning! Your image is being resized")
        print("The current size: (" + str(forced_size[0]) + "," + str(forced_size[1]) + ")")

    #Load all filename
    image_filenames = numpy.array(os.listdir(folder_dir))
    image_filenames.sort()

    #Calculating dimensions
    no_samples = len(image_filenames)
    
    #Defining variables
    sample_pool = numpy.zeros((no_samples, forced_size[0], forced_size[1], 3))

    #Reading all images in the pool
    for i in range(no_samples):

        #Read images from pool
        filename = folder_dir + "/" + image_filenames[i]
        sample_pool[i, :, :, :] = read_single_image(filename=filename, forced_size=forced_size, scaled=scaled)

    #Return pool
    return sample_pool


#Function to read label from folder
def read_labels_from_folder(folder_dir=None, forced_size=None, scaled=True):
    
    #Display
    if(forced_size!=None):
        print("Warning! Your label is being resized")
        print("The current size: (" + str(forced_size[0]) + "," + str(forced_size[1]) + ")")

    #Load all filename
    image_filenames = numpy.array(os.listdir(folder_dir))
    image_filenames.sort()

    #Calculating dimensions
    no_samples = len(image_filenames)

    #Defining variables
    sample_pool = numpy.zeros((no_samples, forced_size[0], forced_size[1], 1))
    
    #Reading all images in the pool
    for i in range(no_samples):

        #Read images from pool
        filename = folder_dir + "/" + image_filenames[i]
        sample_pool[i, :, :, 0] = read_single_label(filename=filename, forced_size=forced_size, scaled=scaled)

    #Return pool
    return sample_pool

