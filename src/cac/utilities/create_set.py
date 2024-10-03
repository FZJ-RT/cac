#-----------------------------------------------------
#Import external libraries
import numpy
import pandas

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#-----------------------------------------------------


#Function to create training and test set
#References:
#This code is adopted from N.Chlis (2018): https://github.com/nchlis/keras_UNET_segmentation/blob/master/unet_train.py
def create_set(pool_image_array=None, pool_label_array=None,
               no_training_samples=None, no_test_samples=None,
               documentation_path=None,
               random_seed=1234567):
    
    #Note on no_test_samples: if user fills this, we split the test sample to get validation sample

    #Error check
    if pool_image_array.shape[0] != pool_label_array.shape[0]:
        err_msg = "You need to provide equal number of images and labels"
        raise ValueError(err_msg)
    
    #Define index
    all_index = numpy.arange(pool_image_array.shape[0])
    
    if(no_test_samples==None):
        
        #Perform splitting
        flag = 0
        while(flag==0):
            
            #Random assignment
            index_training, index_test = train_test_split(all_index, train_size=no_training_samples, random_state=random_seed)
            
            #Check point
            if(len(numpy.intersect1d(index_training, index_test))==0):
                flag = 1
        
        #Assigning to variables
        training_images = pool_image_array[index_training, :]
        training_labels = pool_label_array[index_training, :]

        test_images = pool_image_array[index_test, :]
        test_labels = pool_label_array[index_test, :]
        
        #Documentation
        fnames_training  = index_training.tolist()
        fnames_test      = index_test.tolist()

        fname_split = ['training']*len(fnames_training) + ['test']*len(fnames_test)
        df = pandas.DataFrame({'dataset':fnames_training + fnames_test,
                    'split':fname_split})
        df.to_csv(documentation_path + '/training_test_splits.csv',index=False)

        return training_images, training_labels, test_images, test_labels
    
    else:

        #Perform splitting
        flag = 0
        while(flag==0):
            
            #Random assignment
            index_training, its = train_test_split(all_index, train_size=no_training_samples, random_state=random_seed)
            index_test, index_validation = train_test_split(its, train_size=no_test_samples, random_state=random_seed)
            
            #Check point
            if(len(numpy.intersect1d(index_training, index_test))==0)&(len(numpy.intersect1d(index_training, index_validation))==0)&(len(numpy.intersect1d(index_test, index_validation))==0):
                flag = 1
        
        #Assigning to variables
        training_images = pool_image_array[index_training, :]
        training_labels = pool_label_array[index_training, :]

        test_images = pool_image_array[index_test, :]
        test_labels = pool_label_array[index_test, :]

        prediction_images = pool_image_array[index_validation, :]
        prediction_labels = pool_label_array[index_validation, :]

        #Documentation
        fnames_training   = index_training.tolist()
        fnames_test       = index_test.tolist()
        fnames_prediction = index_validation.tolist()

        fname_split = ['training']*len(fnames_training) + ['test']*len(fnames_test) + ['prediction']*len(fnames_prediction)
        df = pandas.DataFrame({'dataset':fnames_training + fnames_test + fnames_prediction,
                    'split':fname_split})
        df.to_csv(documentation_path + '/training_test_prediction_splits.csv',index=False)

        return training_images, training_labels, test_images, test_labels, prediction_images, prediction_labels