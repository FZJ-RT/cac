#-----------------------------------------------------
#Import external libraries
import os
import shutil
import numpy
import tensorflow
import horovod.tensorflow.keras as hvd
import keras
import bohb.configspace as cs
import time as tmm
import glob
import pickle

from bohb import BOHB
from pathlib import Path
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from keras.models import save_model
from keras.models import load_model
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


#Function to perform hyperparameter optimization
class NIRB():

    save_path = []

    def define_set(self,training_input_array=None, training_output_array=None,
                   test_input_array=None, test_output_array=None):

        #Import internal libraries
        from ..utilities import create_set_nirb
        
        if(training_output_array.ndim < 3):
            self.scaled_training_inputs, self.scaled_training_outputs, \
            self.scaled_test_inputs, self.scaled_test_outputs = create_set_nirb(training_input_array=training_input_array, training_output_array=training_output_array,
                                                                                test_input_array=test_input_array, test_output_array=test_output_array,
                                                                                save_path=self.save_path)
            self.arranged_training_outputs = self.scaled_training_outputs
            self.arranged_test_outputs = self.scaled_test_outputs

            self.test_output_array = test_output_array
        else:
            self.scaled_training_inputs, self.scaled_training_outputs, self.arranged_training_outputs,\
            self.scaled_test_inputs, self.scaled_test_outputs, self.arranged_test_outputs = create_set_nirb(training_input_array=training_input_array, training_output_array=training_output_array,
                                                                                                            test_input_array=test_input_array, test_output_array=test_output_array,
                                                                                                            save_path=self.save_path)

            Ytest = []
            for k in range(test_output_array.shape[1]):
                for j in range(test_output_array.shape[0]):
                    Ytest.append(test_output_array[j, k, :])
            Ytest = numpy.asarray(Ytest) 
            self.test_output_array = Ytest

        #Load scaling factor
        strr = '/inputs_standard_scaling_factor.npy'
        self.input_scaling_factor = numpy.load(self.save_path + strr)

        strr = '/outputs_minmax_scaling_factor.npy'
        self.output_scaling_factor = numpy.load(self.save_path + strr)


    def reduce_output(self, params_cut_off=None, time_cut_off=None, is_large=False, scaled=False, memory_limit='2GB'):
        
        #Import internal libraries
        from ..model import pod_reduction_spatiotemporal, pod_reduction_temporal, pod_reduction_spatiotemporal_dask, pod_reduction_temporal_dask
        
        if((not(params_cut_off is None))&(not(time_cut_off is None))):
            self.data_driven = 0
            if(is_large==True):
                self.output_basis, output_dimension, information_content = pod_reduction_spatiotemporal_dask(self.scaled_training_outputs, params_cut_off=params_cut_off, memory_limit=memory_limit,
                                                                                           save_path=self.save_path)
            else:
                self.output_basis, output_dimension, information_content = pod_reduction_spatiotemporal(self.scaled_training_outputs, params_cut_off=params_cut_off, time_cut_off=time_cut_off,
                                                                                    save_path=self.save_path)
            self.reduced_outputs = numpy.matmul(self.arranged_training_outputs, self.output_basis.T)
        elif((not(params_cut_off is None))&(time_cut_off is None)):
            self.data_driven = 0
            if(is_large==True):
                self.output_basis, output_dimension, information_content = pod_reduction_temporal_dask(self.scaled_training_outputs, params_cut_off=params_cut_off, memory_limit=memory_limit,
                                                                                     save_path=self.save_path)
            else:
                self.output_basis, output_dimension, information_content = pod_reduction_temporal(self.scaled_training_outputs, params_cut_off=params_cut_off,
                                                                                save_path=self.save_path)
            self.reduced_outputs = numpy.matmul(self.scaled_training_outputs, self.output_basis.T)
        elif((params_cut_off is None)&(not(time_cut_off is None))):
            self.data_driven = 0
            
            #Error check
            err_msg = "If the time is reduced, the parameter should be reduced"
            raise ValueError(err_msg)
        else:
            self.data_driven = 1
            output_dimension = self.scaled_training_outputs.shape[1]
            self.reduced_outputs = self.scaled_training_outputs
        
        #Inner scaling
        if(scaled==True):
            self.inner_scaling = 1

            inner_scaler = preprocessing.StandardScaler().fit(self.reduced_outputs)
            self.reduced_outputs = inner_scaler.transform(self.reduced_outputs)

            scaling_factor = []
            scaling_factor.append(inner_scaler.mean_)
            scaling_factor.append(inner_scaler.var_)
            self.inner_scaling_factor = numpy.asarray(scaling_factor)
            
            numpy.save(self.save_path + "/inner_standard_scaling_factor", self.inner_scaling_factor)
        else:
            self.inner_scaling = 0
        
        numpy.save(self.save_path + "/information_content", information_content)
        print("Output dimension: " + str(output_dimension))
    

    def execute(self, ml_type='nn',
                neurons_in_layer1=None,
                neurons_in_layer2=None,
                branch_neurons_in_layer1=None,
                branch_neurons_in_layer2=None,
                trunk_neurons_in_layer1=None,
                trunk_neurons_in_layer2=None,
                epoch=None,
                learning_rate=None,
                no_length_scales=None,
                activation_function='sigmoid',
                branch_activation_function='sigmoid',
                trunk_activation_function='sigmoid',
                optimization_steps=30,
                loss_function='mse',
                distributed=False):
        
        #Defining gpu
        self.distributed = distributed

        #Defining model type
        self.ml_type = ml_type

        #Transferring loss function
        self.loss_function = loss_function
        
        #Transfering activation function
        self.activation_function = activation_function

        #Create experiments folder
        str1 = "/experiments"
        self.experiment_path = self.save_path + str1
        if(os.path.isdir(self.experiment_path)):
            shutil.rmtree(self.experiment_path)
            os.makedirs(self.experiment_path)
        else:
            os.makedirs(self.experiment_path)
        
        #define search space
        if(self.ml_type=='gpr'):
            self.no_length_scales = no_length_scales
            ll = []
            for ww in range(self.no_length_scales):
                ll.append(cs.UniformHyperparameter('l' + str(ww), lower=1e-5, upper=1e5, log=True))
            ll.append(cs.UniformHyperparameter('lnu', lower=0.5, upper=2.5, log=False))
            configspace = cs.ConfigurationSpace(ll, 
                                                seed=123456) 
        elif(self.ml_type=='nn'):
            neurons_l1 = cs.IntegerUniformHyperparameter("neurons_l1", lower=neurons_in_layer1[0], upper=neurons_in_layer1[1])
            neurons_l2 = cs.IntegerUniformHyperparameter("neurons_l2", lower=neurons_in_layer2[0], upper=neurons_in_layer2[1])
            epoch = cs.IntegerUniformHyperparameter("epoch", lower=epoch[0], upper=epoch[1])
            lr = cs.UniformHyperparameter('lr', lower=learning_rate[0], upper=learning_rate[1], log=True)
            batch = cs.IntegerUniformHyperparameter("batch", lower=1, upper=self.scaled_training_inputs.shape[0])
    
            configspace = cs.ConfigurationSpace([neurons_l1, 
                                                neurons_l2, 
                                                epoch, 
                                                lr, 
                                                batch], 
                                                seed=123456) 
        elif(self.ml_type=='resnet'):
            neurons_l1 = cs.IntegerUniformHyperparameter("neurons_l1", lower=neurons_in_layer1[0], upper=neurons_in_layer1[1])
            neurons_l2 = cs.IntegerUniformHyperparameter("neurons_l2", lower=neurons_in_layer2[0], upper=neurons_in_layer2[1])
            epoch = cs.IntegerUniformHyperparameter("epoch", lower=epoch[0], upper=epoch[1])
            lr = cs.UniformHyperparameter('lr', lower=learning_rate[0], upper=learning_rate[1], log=True)
            batch = cs.IntegerUniformHyperparameter("batch", lower=1, upper=self.scaled_training_inputs.shape[0])
    
            configspace = cs.ConfigurationSpace([neurons_l1, 
                                                neurons_l2, 
                                                epoch, 
                                                lr, 
                                                batch], 
                                                seed=123456) 
        elif(self.ml_type=='bt'):
            branch_neurons_l1 = cs.IntegerUniformHyperparameter("branch_neurons_l1", lower=branch_neurons_in_layer1[0], upper=branch_neurons_in_layer1[1])
            branch_neurons_l2 = cs.IntegerUniformHyperparameter("branch_neurons_l2", lower=branch_neurons_in_layer2[0], upper=branch_neurons_in_layer2[1])
            trunk_neurons_l1 = cs.IntegerUniformHyperparameter("trunk_neurons_l1", lower=trunk_neurons_in_layer1[0], upper=trunk_neurons_in_layer1[1])
            trunk_neurons_l2 = cs.IntegerUniformHyperparameter("trunk_neurons_l2", lower=trunk_neurons_in_layer2[0], upper=trunk_neurons_in_layer2[1])
            epoch = cs.IntegerUniformHyperparameter("epoch", lower=epoch[0], upper=epoch[1])
            lr = cs.UniformHyperparameter('lr', lower=learning_rate[0], upper=learning_rate[1], log=True)
            batch = cs.IntegerUniformHyperparameter("batch", lower=1, upper=self.scaled_training_inputs.shape[0])
    
            configspace = cs.ConfigurationSpace([branch_neurons_l1, 
                                                branch_neurons_l2, 
                                                trunk_neurons_l1, 
                                                trunk_neurons_l2,
                                                epoch, 
                                                lr, 
                                                batch], 
                                                seed=123456) 


        def evaluate(params, budget):
            inputs = numpy.array(list(params.items()))[:,1]
            loss = self.machine_learning_model(inputs)
            return loss 
        
        #Start timer
        start = tmm.time() 

        print("\nBegin optimization: ")

        opt = BOHB(configspace, evaluate, max_budget=optimization_steps, min_budget=1, n_proc=1)
        logs = opt.optimize()
        
        print("End optimization")

        #End timer
        end = tmm.time()
        self.timecalc = end-start

        #Collect the results
        txtfiles = []
        for file in glob.glob(self.experiment_path + "/*_trained_model.npy"):
            txtfiles.append(file)
        
        losses = []
        for kk in range(len(txtfiles)):
            losses.extend([numpy.load(txtfiles[kk])[-1]])
        losses = numpy.asarray(losses)
        index_min = numpy.argmin(numpy.asarray(losses))

        #Loading model and reading BOHB parameters
        if(self.ml_type=='gpr'):
            str1 = self.experiment_path + "/" + str(round(numpy.asarray(losses)[index_min], 8)) + "_trained_model.pickle"
            with open(str1, 'rb') as file:
                networks = pickle.load(file)
        else:
            str1 = self.experiment_path + "/" + str(round(numpy.asarray(losses)[index_min], 8)) + "_trained_model.tf"
            networks = load_model(str1, compile=False)
        str2 = self.experiment_path + "/" + str(round(numpy.asarray(losses)[index_min], 8)) + "_trained_model.npy"
        hyperparameters = numpy.load(str2)

        #Re-saving model for better naming
        if(self.ml_type=='gpr'):
            strr = self.save_path + "/trained_model.pickle"
            with open(strr, 'wb') as file:
                pickle.dump(networks, file)
        else:
            strr = self.save_path + "/trained_model.tf"
            save_model(networks,strr)
        numpy.save(os.path.join(self.save_path, "trained_model_hyperparameters.npy"), hyperparameters) 
        
        #Calculating MSE_validation full solution with PoD scaling
        if(self.ml_type=='gpr'):
            pred_test_output = networks.predict(self.scaled_test_inputs, return_std=False, return_cov=False) #RB coefficients unscaled, prediction
        elif(self.ml_type=='resnet'):
            pred_test_output = networks(self.scaled_test_inputs, training=False) #RB coefficients unscaled, prediction
        elif(self.ml_type=='nn'):
            pred_test_output = networks(self.scaled_test_inputs, training=False) #RB coefficients unscaled, prediction
        elif(self.ml_type=='bt'):
            pred_test_output = networks([self.scaled_test_inputs[:,1:], self.scaled_test_inputs[:,0]], training=False) #RB coefficients unscaled, prediction

        #Inner scaling
        if(self.inner_scaling==1):
            temp = numpy.zeros( (pred_test_output.shape[0],pred_test_output.shape[1]) )
            for i in range(pred_test_output.shape[1]):
                if(self.inner_scaling_factor[1, i]==0):
                    temp[:,i] = 0.0
                else:
                    temp[:,i] = (pred_test_output[:,i] * numpy.sqrt(self.inner_scaling_factor[1, i])) + self.inner_scaling_factor[0, i]
            pred_test_output = temp
        
        if(self.data_driven==0):
            scaled_output = numpy.dot(pred_test_output, self.output_basis)
        elif(self.data_driven==1):
            scaled_output = pred_test_output

        unscaled_output = numpy.zeros( (scaled_output.shape[0], scaled_output.shape[1]) )
        for i in range(scaled_output.shape[1]):
            if((self.output_scaling_factor[1, i] - self.output_scaling_factor[0, i])==0):
                unscaled_output[:,i] = 0.0
            else:
                unscaled_output[:,i] = (scaled_output[:,i] * (self.output_scaling_factor[1, i] - self.output_scaling_factor[0, i])) + self.output_scaling_factor[0, i]
 
        if(self.loss_function=="mse"):
            test_error = root_mean_squared_error(self.test_output_array, unscaled_output)  
        else:
            test_error = mean_absolute_error(self.test_output_array, unscaled_output)       
        
        #Tracking training parameters
        params = []
        for i in range(numpy.asarray(numpy.asarray(losses).shape)[0]):
            str2 = self.experiment_path + "/" + str(round(numpy.asarray(losses)[i], 8)) + "_trained_model.npy"
            params.append(numpy.load(str2))
        
        if(self.ml_type=='gpr'):
            for f in Path(self.save_path).glob('*_trained_model.pickle'):
                f.unlink()
        else:
            for f in Path(self.save_path).glob('*_trained_model.tf'):
                f.unlink()
        
        for f in Path(self.save_path).glob('*_trained_model.npy'):
            f.unlink()

        os.system("rm -r " + self.experiment_path)
        
        #Display
        if(self.loss_function=="mse"):
            print("Final L2-norm for full solutions: " + str(round(test_error, 8)))
        else:
            print("Final L1-norm for full solutions: " + str(round(test_error, 8)))


    def machine_learning_model(self, config):
        
        #Assigning parameters
        if(self.ml_type=='gpr'):
            ll = []
            for ww in range(self.no_length_scales):
                ll.extend([float(config[ww])])
        elif(self.ml_type=='nn'):
            number_neurons_l1 = int(config[0])
            number_neurons_l2 = int(config[1])
            number_epochs = int(config[2])
            lrr = float(config[3])
            batcht = int(config[4])
        elif(self.ml_type=='resnet'):
            number_neurons_l1 = int(config[0])
            number_neurons_l2 = int(config[1])
            number_epochs = int(config[2])
            lrr = float(config[3])
            batcht = int(config[4])
        elif(self.ml_type=='bt'):
            branch_number_neurons_l1 = int(config[0])
            branch_number_neurons_l2 = int(config[1])
            trunk_number_neurons_l1 = int(config[2])
            trunk_number_neurons_l2 = int(config[3])
            number_epochs = int(config[4])
            lrr = float(config[5])
            batcht = int(config[6])

        #Start timer
        start = tmm.time() 

        #Constructing architecture
        from ..model import create_nirb, create_nirb_resnet, create_nirb_gpr, create_nirb_bt

        if(self.ml_type=='nn'):
            nirb = create_nirb(self.scaled_training_inputs, self.reduced_outputs, 
                            neurons_in_layer1=number_neurons_l1,
                            neurons_in_layer2=number_neurons_l2, 
                            learning_rate=lrr,
                            activation_function=self.activation_function,
                            loss_function=self.loss_function,
                            distributed=self.distributed)
        elif(self.ml_type=='resnet'):
            nirb = create_nirb_resnet(self.scaled_training_inputs, self.reduced_outputs, 
                                      neurons_in_layer1=number_neurons_l1,
                                      neurons_in_layer2=number_neurons_l2, 
                                      learning_rate=lrr,
                                      loss_function=self.loss_function,
                                      distributed=self.distributed)
        elif(self.ml_type=='gpr'):
            trained_nirb = create_nirb_gpr(self.scaled_training_inputs, self.reduced_outputs,
                                           ll,
                                           float(config[-1]))
        elif(self.ml_type=='bt'):
            nirb = create_nirb_bt(self.scaled_training_inputs[:,1:], self.scaled_training_inputs[:,0], self.reduced_outputs, 
                                  branch_neurons_in_layer1=branch_number_neurons_l1, branch_neurons_in_layer2=branch_number_neurons_l2, 
                                  trunk_neurons_in_layer1=trunk_number_neurons_l1, trunk_neurons_in_layer2=trunk_number_neurons_l2, 
                                  loss_function=self.loss_function, learning_rate=lrr,
                                  distributed=self.distributed)
        
        #Inner training
        if(self.ml_type=='nn'):
            from ..trainer import train_regression_without_generator

            trained_nirb = train_regression_without_generator(self.scaled_training_inputs, self.reduced_outputs, 
                                                            nirb, 
                                                            batch_size=batcht, 
                                                            epochs=number_epochs,
                                                            distributed=self.distributed)
        elif(self.ml_type=='resnet'):
            from ..trainer import train_regression_without_generator

            trained_nirb = train_regression_without_generator(self.scaled_training_inputs, self.reduced_outputs, 
                                                            nirb, 
                                                            batch_size=batcht, 
                                                            epochs=number_epochs,
                                                            distributed=self.distributed)
        elif(self.ml_type=='bt'):
            from ..trainer import train_regression_without_generator_bt

            trained_nirb = train_regression_without_generator_bt(self.scaled_training_inputs, self.reduced_outputs, 
                                                                 nirb, 
                                                                 batch_size=batcht, 
                                                                 epochs=number_epochs,
                                                                 distributed=self.distributed)

        #End timer
        end = tmm.time()
        timecalc = end-start

        #ML prediction for RB coefficients
        if(self.ml_type=='gpr'):
            pred_test_output = trained_nirb.predict(self.scaled_test_inputs, return_std=False, return_cov=False) #RB coefficients
        elif(self.ml_type=='resnet'):
            pred_test_output = trained_nirb(self.scaled_test_inputs, training=False) #RB coefficients
        elif(self.ml_type=='nn'):
            pred_test_output = trained_nirb(self.scaled_test_inputs, training=False) #RB coefficients
        elif(self.ml_type=='bt'):
            pred_test_output = trained_nirb([self.scaled_test_inputs[:,1:], self.scaled_test_inputs[:,0]], training=False) #RB coefficients

        #Inner scaling
        if(self.inner_scaling==1):
            temp = numpy.zeros( (pred_test_output.shape[0],pred_test_output.shape[1]) )
            for i in range(pred_test_output.shape[1]):
                if(self.inner_scaling_factor[1, i]==0):
                    temp[:,i] = 0.0
                else:
                    temp[:,i] = (pred_test_output[:,i] * numpy.sqrt(self.inner_scaling_factor[1, i])) + self.inner_scaling_factor[0, i]
            pred_test_output = temp

        #Full reconstruction PoD_scaled
        if(self.data_driven==0):
            if(self.loss_function=="mse"):
                test_error = mean_squared_error(self.arranged_test_outputs, numpy.dot(pred_test_output, self.output_basis))  
            else:
                test_error = mean_absolute_error(self.arranged_test_outputs, numpy.dot(pred_test_output, self.output_basis))      
        elif(self.data_driven==1): 
            if(self.loss_function=="mse"):
                test_error = mean_squared_error(self.arranged_test_outputs, pred_test_output)  
            else:
                test_error = mean_absolute_error(self.arranged_test_outputs, pred_test_output)      

        #Saving model
        if(self.ml_type=='gpr'):
            strr = self.experiment_path + "/" + str(round(test_error, 8)) + "_trained_model.pickle"
            with open(strr, 'wb') as file:
                pickle.dump(trained_nirb, file)
            ll.extend([float(config[-1])])
            ll.extend([timecalc])
            ll.extend([test_error])
            params = ll
        elif(self.ml_type=='resnet'):
            strr = self.experiment_path + "/" + str(round(test_error, 8)) + "_trained_model.tf"
            save_model(trained_nirb,strr)
            params = [number_neurons_l1, number_neurons_l2, number_epochs, lrr, batcht, timecalc, test_error]
        elif(self.ml_type=='nn'):
            strr = self.experiment_path + "/" + str(round(test_error, 8)) + "_trained_model.tf"
            save_model(trained_nirb,strr)
            params = [number_neurons_l1, number_neurons_l2, number_epochs, lrr, batcht, timecalc, test_error]
        elif(self.ml_type=='bt'):
            strr = self.experiment_path + "/" + str(round(test_error, 8)) + "_trained_model.tf"
            save_model(trained_nirb,strr)
            params = [branch_number_neurons_l1, branch_number_neurons_l2, trunk_number_neurons_l1, trunk_number_neurons_l2, number_epochs, lrr, batcht, timecalc, test_error]

        numpy.save(os.path.join(self.experiment_path, str(round(test_error, 8)) + "_trained_model.npy"), params)
        print("Current error: " + str(round(test_error, 8)))

        tmm.sleep(0.1)
        return test_error