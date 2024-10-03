#-----------------------------------------------------
#Import external libraries
import numpy

#-----------------------------------------------------


#Function to perform superposition
def superpose_reference(current_label, reference_label):
    
    #Error check
    if(current_label.shape[0]!=reference_label.shape[0])|(current_label.shape[1]!=reference_label.shape[1]):
        err_msg = "You do not provide the same size for current label and reference label"
        raise ValueError(err_msg)

    #Perform superposition
    for yy in range(reference_label.shape[0]):
        for xx in range(reference_label.shape[1]):
            if(reference_label[yy,xx]==0):
                current_label[yy,xx] = 0

    #Return current label
    return current_label

