#-----------------------------------------------------
#Import external libraries
import numpy
import skimage.transform

#References:
#This code is adopted from N.Chlis (2018): https://github.com/nchlis/keras_UNET_segmentation/blob/master/unet_train.py
#-----------------------------------------------------


#Function to perform rotation
def rotate_image(image_array, rotation_angle):
    
    #Define variables
    transformed_image = numpy.zeros_like(image_array)

    #Perform rotation
    for ch in numpy.arange(image_array.shape[-1]):
        for i in numpy.arange(image_array.shape[0]):
            transformed_image[i,:,:,ch] = skimage.transform.rotate(image_array[i,:,:,ch],angle=rotation_angle,resize=False,preserve_range=True,mode='edge')
    
    #Return transformed images
    return transformed_image


#Function to perform translation -- currently its not used
def translate_image(image_array, dx, dy):
    
    #Define variables
    transformed_image = numpy.zeros_like(image_array)

    #Perform rotation
    for i in numpy.arange(image_array.shape[0]):
        transformed_image[i,:,:,:] = skimage.transform.warp(image_array[i,:,:,:],skimage.transform.SimilarityTransform(translation=(dx, dy)),mode='edge')
    
    #Return transformed images
    return transformed_image


#Function to create augmentation
def augmentation_generator(X=None,Y=None,
                           batch_size=4,
                           flip_axes=['x','y'],
                           rotation_angles=[5,15]):
    
    #Error check
    if X.ndim < 4:
        err_msg = "You need to provide more samples"
        raise ValueError(err_msg)
    if Y.ndim < 4:
        err_msg = "It has to be RGB"
        raise ValueError(err_msg)

    #Calculate total number of samples
    no_samples = len(X)

    while(True):
        #Shuffling samples into batches
        ix_randomized = numpy.random.choice(no_samples, size=no_samples, replace=False)

        #Calculate number of batches
        ix_batches = numpy.array_split(ix_randomized, int(no_samples/batch_size))

        #Create augmenttaion for each batch
        for b in range(len(ix_batches)):
            
            #Prepare for batch looping
            ix_batch = ix_batches[b]
            current_batch_size=len(ix_batch)
            
            #Copy the images to a secondary variables
            X_batch = X[ix_batch,:,:,:].copy()
            Y_batch = Y[ix_batch,:,:,:].copy()
            
            #Looping to flix the images
            for img in range(current_batch_size):
                
                #Throwing random numbers
                do_aug=numpy.random.choice([True, False],size=1)[0]
                
                #Performing random flipping
                if do_aug == True:

                    #Randomly select rotation axis
                    flip_axis_selected = numpy.random.choice(flip_axes,1,replace=False)[0]
                    if flip_axis_selected == 'x':
                        flip_axis_selected = 1
                    else: 
                        flip_axis_selected = 0
                    
                    #Flip images
                    X_batch[img,:,:,:] = numpy.flip(X_batch[img,:,:,:],axis=flip_axis_selected)
                    Y_batch[img,:,:,:] = numpy.flip(Y_batch[img,:,:,:],axis=flip_axis_selected)
                
                #Throwing random numbers
                do_aug=numpy.random.choice([True, False],size=1)[0]

                #Perfoming random rotation
                if do_aug == True:
                    
                    #Randomly select rotation angles
                    rotation_angle_selected = numpy.random.uniform(low=rotation_angles[0],high=rotation_angles[1],size=1)[0]
                    
                    #Rotate images
                    X_batch[img,:,:,:] = rotate_image(numpy.expand_dims(X_batch[img,:,:,:],axis=0), rotation_angle=rotation_angle_selected)
                    Y_batch[img,:,:,:] = rotate_image(numpy.expand_dims(Y_batch[img,:,:,:],axis=0), rotation_angle=rotation_angle_selected)
        
        #Return but it is for generator
        yield(X_batch,Y_batch)


