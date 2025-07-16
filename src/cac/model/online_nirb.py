#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import keras
import os
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


#Function to perform online phase
class online_nirb():
    
    
    def __init__(self, ml_type, save_path):
        
        self.ml_type      = ml_type
        self.save_path    = save_path
        self.model        = None
        self.input_basis  = None
        self.output_basis = None

      
    def load_existing_model(self):
        
        if(self.ml_type=='gpr'):
            strr = '/trained_model.pickle'
            with open(self.save_path + strr, 'rb') as file:
                self.model = pickle.load(file)
        else:
            strr = '/trained_model.tf'
            self.model = load_model(self.save_path + strr, compile=False)
        print(self.model.summary())

    
    def load_input_basis(self):

        strr = '/input_basis.npy'
        self.input_basis = numpy.load(self.save_path + strr)

    
    def load_output_basis(self, data_driven=False):
        
        if(data_driven==False):
            self.data_driven = 0
            strr = '/output_basis.npy'
            self.output_basis = numpy.load(self.save_path + strr)
        else:
            self.data_driven = 1

    
    def load_scaling_factor(self, inner_scaled=False):

        strr = '/inputs_standard_scaling_factor.npy'
        self.input_scaling_factor = numpy.load(self.save_path + strr)

        strr = '/outputs_minmax_scaling_factor.npy'
        self.output_scaling_factor = numpy.load(self.save_path + strr)
        
        if(inner_scaled==True):
            self.inner_scaling = 1

            strr = '/inner_standard_scaling_factor.npy'
            self.inner_scaling_factor = numpy.load(self.save_path + strr)
        else:
            self.inner_scaling = 0

    
    def load_test_set(self, test_input_array, test_output_array, criterion='l2_norm'):

        #Scaling test inputs
        scaled_test_inputs = numpy.zeros((test_input_array.shape[0], test_input_array.shape[1]))
        for i in range(test_input_array.shape[1]):
            if(self.input_scaling_factor[1, i]==0):
                scaled_test_inputs[:,i] = 0.0
            else:
                scaled_test_inputs[:,i] = (test_input_array[:,i] - self.input_scaling_factor[0, i])/numpy.sqrt(self.input_scaling_factor[1, i])

        #Check for input reduction
        if(not(self.input_basis is None)):
            reduced_scaled_test_inputs = numpy.matmul(scaled_test_inputs, self.input_basis.T)
        else:
            reduced_scaled_test_inputs = scaled_test_inputs

        #Prediction
        if(self.ml_type=='gpr'):
            predicted_reduced_scaled_test_outputs = self.model.predict(reduced_scaled_test_inputs, return_std=False, return_cov=False)
        elif(self.ml_type=='resnet'):
            predicted_reduced_scaled_test_outputs = self.model(reduced_scaled_test_inputs, training=False)
        elif(self.ml_type=='nn'):
            predicted_reduced_scaled_test_outputs = self.model(reduced_scaled_test_inputs, training=False)
        elif(self.ml_type=='bt'):
            predicted_reduced_scaled_test_outputs = self.model([reduced_scaled_test_inputs[:,1:], reduced_scaled_test_inputs[:,0]], training=False)
        
        #Inner scaling
        if(self.inner_scaling==1):
            temp = numpy.zeros( (predicted_reduced_scaled_test_outputs.shape[0],predicted_reduced_scaled_test_outputs.shape[1]) )
            for i in range(predicted_reduced_scaled_test_outputs.shape[1]):
                if(self.inner_scaling_factor[1, i]==0):
                    temp[:,i] = 0.0
                else:
                    temp[:,i] = (predicted_reduced_scaled_test_outputs[:,i] * numpy.sqrt(self.inner_scaling_factor[1, i])) + self.inner_scaling_factor[0, i]
            predicted_reduced_scaled_test_outputs = temp

        #Put back to real space
        if(self.data_driven==0):
            predicted_scaled_test_outputs =  numpy.dot(predicted_reduced_scaled_test_outputs, self.output_basis)
        else:
            predicted_scaled_test_outputs = predicted_reduced_scaled_test_outputs

        #Rescale back
        predicted_arranged_test_outputs = numpy.zeros( (predicted_scaled_test_outputs.shape[0], predicted_scaled_test_outputs.shape[1]) )
        for i in range(predicted_scaled_test_outputs.shape[1]):
            if((self.output_scaling_factor[1, i] - self.output_scaling_factor[0, i])==0):
                predicted_arranged_test_outputs[:,i] = 0.0
            else:
                predicted_arranged_test_outputs[:,i] = (predicted_scaled_test_outputs[:,i] * (self.output_scaling_factor[1, i] - self.output_scaling_factor[0, i])) + self.output_scaling_factor[0, i]
        
        #Re-arrange outputs
        arranged_test_outputs = []
        for k in range(test_output_array.shape[1]):
            for j in range(test_output_array.shape[0]):
                arranged_test_outputs.append(test_output_array[j, k, :])
        arranged_test_outputs = numpy.asarray(arranged_test_outputs) 

        #Calculate error
        if(criterion=="l2_norm"):
            test_error =  mean_squared_error(arranged_test_outputs, predicted_arranged_test_outputs, squared=False)
        elif(criterion=="l1_norm"):
            test_error = mean_absolute_error(arranged_test_outputs, predicted_arranged_test_outputs)
        else:
            err_msg = "Error criterion is not recognized"
            raise ValueError(err_msg)

        #Return values
        return test_error


    def predict(self, prediction_input_array, prediction_output_array=None, criterion='l2_norm'):

        #Scaling prediction inputs
        scaled_prediction_inputs = numpy.zeros((prediction_input_array.shape[0], prediction_input_array.shape[1]))
        for i in range(prediction_input_array.shape[1]):
            if(self.input_scaling_factor[1, i]==0):
                scaled_prediction_inputs[:,i] = 0.0
            else:
                scaled_prediction_inputs[:,i] = (prediction_input_array[:,i] - self.input_scaling_factor[0, i])/numpy.sqrt(self.input_scaling_factor[1, i])
        
        #Check for input reduction
        if(not(self.input_basis is None)):
            reduced_scaled_prediction_inputs = numpy.matmul(scaled_prediction_inputs, self.input_basis.T)
        else:
            reduced_scaled_prediction_inputs = scaled_prediction_inputs

        #Prediction
        if(self.ml_type=='gpr'):
            predicted_reduced_scaled_prediction_outputs = self.model.predict(reduced_scaled_prediction_inputs, return_std=False, return_cov=False)
        elif(self.ml_type=='resnet'):
            predicted_reduced_scaled_prediction_outputs = self.model(reduced_scaled_prediction_inputs, training=False)
        elif(self.ml_type=='nn'):
            predicted_reduced_scaled_prediction_outputs = self.model(reduced_scaled_prediction_inputs, training=False)
        elif(self.ml_type=='bt'):
            predicted_reduced_scaled_prediction_outputs = self.model([reduced_scaled_prediction_inputs[:,1:],reduced_scaled_prediction_inputs[:,0]], training=False)

        #Inner scaling
        if(self.inner_scaling==1):
            temp = numpy.zeros( (predicted_reduced_scaled_prediction_outputs.shape[0],predicted_reduced_scaled_prediction_outputs.shape[1]) )
            for i in range(predicted_reduced_scaled_prediction_outputs.shape[1]):
                if(self.inner_scaling_factor[1, i]==0):
                    temp[:,i] = 0.0
                else:
                    temp[:,i] = (predicted_reduced_scaled_prediction_outputs[:,i] * numpy.sqrt(self.inner_scaling_factor[1, i])) + self.inner_scaling_factor[0, i]
            predicted_reduced_scaled_prediction_outputs = temp

        #Put back to real space
        if(self.data_driven==0):
            predicted_scaled_prediction_outputs =  numpy.dot(predicted_reduced_scaled_prediction_outputs, self.output_basis)
        else:
            predicted_scaled_prediction_outputs = predicted_reduced_scaled_prediction_outputs
            
        #Rescale back
        predicted_arranged_prediction_outputs = numpy.zeros( (predicted_scaled_prediction_outputs.shape[0], predicted_scaled_prediction_outputs.shape[1]) )
        for i in range(predicted_scaled_prediction_outputs.shape[1]):
            if((self.output_scaling_factor[1, i] - self.output_scaling_factor[0, i])==0):
                predicted_arranged_prediction_outputs[:,i] = 0.0
            else:
                predicted_arranged_prediction_outputs[:,i] = (predicted_scaled_prediction_outputs[:,i] * (self.output_scaling_factor[1, i] - self.output_scaling_factor[0, i])) + self.output_scaling_factor[0, i]
        
        #Check for comparison
        if(prediction_output_array is None):

            return predicted_arranged_prediction_outputs

        else:

            #Re-arrange outputs
            arranged_prediction_outputs = []
            for k in range(prediction_output_array.shape[1]):
                for j in range(prediction_output_array.shape[0]):
                    arranged_prediction_outputs.append(prediction_output_array[j, k, :])
            arranged_prediction_outputs = numpy.asarray(arranged_prediction_outputs) 

            #Calculate error
            if(criterion=="l2_norm"):
                prediction_error =  mean_squared_error(arranged_prediction_outputs, predicted_arranged_prediction_outputs, squared=False)
            elif(criterion=="l1_norm"):
                prediction_error = mean_absolute_error(arranged_prediction_outputs, predicted_arranged_prediction_outputs)
            else:
                err_msg = "Error criterion is not recognized"
                raise ValueError(err_msg)

            #Return values
            return predicted_arranged_prediction_outputs, prediction_error