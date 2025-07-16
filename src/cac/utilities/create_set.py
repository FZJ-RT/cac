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


#Function to create training and test set for nirb
def create_set_nirb(training_input_array=None, training_output_array=None,
                    test_input_array=None, test_output_array=None,
                    save_path=None):
    
    #Scaling inputs
    inputs_scaler = preprocessing.StandardScaler().fit(training_input_array)
    scaled_training_input_array = inputs_scaler.transform(training_input_array)
    scaled_test_input_array     = inputs_scaler.transform(test_input_array)

    scaling_factor = []
    scaling_factor.append(inputs_scaler.mean_)
    scaling_factor.append(inputs_scaler.var_)
    scaling_factor = numpy.asarray(scaling_factor)
    
    numpy.save(save_path + "/inputs_standard_scaling_factor", scaling_factor)


    #Scaling outputs
    if(training_output_array.ndim < 3):
        outputs_scaler = preprocessing.MinMaxScaler().fit(training_output_array)
        scaled_training_output_array = outputs_scaler.transform(training_output_array)
        scaled_test_output_array     = outputs_scaler.transform(test_output_array)
        
        scaling_factor = []
        scaling_factor.append(outputs_scaler.data_min_)
        scaling_factor.append(outputs_scaler.data_max_)
        scaling_factor = numpy.asarray(scaling_factor)

        numpy.save(save_path + "/outputs_minmax_scaling_factor", scaling_factor)
        
        #Return set
        return scaled_training_input_array, scaled_training_output_array, scaled_test_input_array, scaled_test_output_array 
    else:
        Ytrain = []
        for k in range(training_output_array.shape[1]):
            for j in range(training_output_array.shape[0]):
                Ytrain.append(training_output_array[j, k, :])
        Ytrain = numpy.asarray(Ytrain) 

        Ytest = []
        for k in range(test_output_array.shape[1]):
            for j in range(test_output_array.shape[0]):
                Ytest.append(test_output_array[j, k, :])
        Ytest = numpy.asarray(Ytest) 
        
        outputs_scaler = preprocessing.MinMaxScaler().fit(Ytrain)
        scaled_Ytrain  = outputs_scaler.transform(Ytrain)
        scaled_Ytest   = outputs_scaler.transform(Ytest)
        
        scaled_training_output_array = numpy.zeros((training_output_array.shape[0], training_output_array.shape[1], training_output_array.shape[2])) 
        scaled_test_output_array     = numpy.zeros((test_output_array.shape[0], test_output_array.shape[1], test_output_array.shape[2])) 
        
        aa = 0
        for k in range(training_output_array.shape[1]):
            for j in range(training_output_array.shape[0]):
                scaled_training_output_array[j, k, :] = scaled_Ytrain[aa,:]
                aa = aa + 1 

        aa = 0
        for k in range(test_output_array.shape[1]):
            for j in range(test_output_array.shape[0]):
                scaled_test_output_array[j, k, :] = scaled_Ytest[aa,:]
                aa = aa + 1 

        scaling_factor = []
        scaling_factor.append(outputs_scaler.data_min_)
        scaling_factor.append(outputs_scaler.data_max_)
        scaling_factor = numpy.asarray(scaling_factor)

        numpy.save(save_path + "/outputs_minmax_scaling_factor", scaling_factor)

        #Return set
        return scaled_training_input_array, scaled_training_output_array, scaled_Ytrain, \
               scaled_test_input_array, scaled_test_output_array, scaled_Ytest 
