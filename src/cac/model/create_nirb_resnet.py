#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import keras
import horovod.tensorflow.keras as hvd
from keras import Sequential
from keras.layers import Dense, Input
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------

#Define customed bias layer
class BiasLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias


#Function to perform reduction on output side, spatio temporal
def pod_reduction_spatiotemporal(snapshots, params_cut_off=99.999, time_cut_off=99.999,
                                 save_path=None):
    
    #Check dimensions
    Ns = snapshots.shape[0]
    Nt = snapshots.shape[1]

    #Reduction
    basis_matrix = []
    basis_time_matrix = []
    information_content = []

    for i in range(Ns): 
        
        #Enumeration for included eigen vectors
        index = 0

        #Perform parameter svd
        svd = numpy.linalg.svd(snapshots[i, :, :], full_matrices=False)  
        
        #Calculate total eigne values
        sum_eigen_full = numpy.sum(svd[1]**2)

        for j in range(len(svd[1])):
            
            #Calculate time eigen sum
            sum_eigen = numpy.sum(svd[1][:j]**2) 

            #Performing time cut off
            information_content.append(sum_eigen/sum_eigen_full)
            if((sum_eigen/sum_eigen_full)<(time_cut_off/100)):
                index+=1

        basis_time_matrix.extend(numpy.copy(svd[2][:index]))

    #Enumeration for included eigen vectors
    index = 0

    #Perform parameter svd
    svd = numpy.linalg.svd(numpy.asarray(basis_time_matrix), full_matrices=False) 
    
    #Calculate total eigne values
    sum_eigen_full = numpy.sum(svd[1]**2)

    for j in range(len(svd[1])):
        
        #Calculate time eigen sum
        sum_eigen = numpy.sum(svd[1][:j]**2) 

        #Performing parmeters cut off
        information_content.append(sum_eigen/sum_eigen_full)
        if((sum_eigen/sum_eigen_full)<(params_cut_off/100)):
            index+=1

    basis_matrix.extend(numpy.copy(svd[2][:index]))
    
    #Transforming basis matrix to an array
    basis_matrix = numpy.asarray(basis_matrix)
    
    #Save
    if(save_path!=None):
        strr = '/output_basis.npy'
        numpy.save(save_path + strr, basis_matrix)

    #Error check
    if(basis_matrix.shape[0]<=1):
        err_msg = "Basis size is 1. Please reduce cut off!"
        raise ValueError(err_msg)

    #Return
    return basis_matrix, basis_matrix.shape[0], information_content


#Function to perform reduction on output side, temporal
def pod_reduction_temporal(snapshots, params_cut_off=99.999,
                           save_path=None):
    
    #Check dimensions
    Ns = snapshots.shape[0]

    #Reduction
    basis_matrix = []
    basis_time_matrix = []
    information_content = []

    #Enumeration for included eigen vectors
    index = 0

    #Perform parameter svd
    svd = numpy.linalg.svd(snapshots, full_matrices=False) 
    
    #Calculate total eigne values
    sum_eigen_full = numpy.sum(svd[1]**2)

    for j in range(len(svd[1])):
        
        #Calculate time eigen sum
        sum_eigen = numpy.sum(svd[1][:j]**2) 

        #Performing parmeters cut off
        information_content.append(sum_eigen/sum_eigen_full)
        if((sum_eigen/sum_eigen_full)<(params_cut_off/100)):
            index+=1

    basis_matrix = numpy.copy(svd[2][:index])
    
    #Save
    if(save_path!=None):
        strr = '/output_basis.npy'
        numpy.save(save_path + strr, basis_matrix)

    #Error check
    if(basis_matrix.shape[0]<=1):
        err_msg = "Basis size is 1. Please reduce cut off!"
        raise ValueError(err_msg)

    #Return
    return basis_matrix, basis_matrix.shape[0], information_content


#Function to create resnet blocks
def resnet_block(x, no_neurons):
	output_shape = x.shape
	assert len(output_shape) == 2
	output_dim = output_shape[-1]
	intermediate = Dense(no_neurons, activation = 'sigmoid')(x)
	return Dense(output_dim)(intermediate)


#Function to compile all architectures
def compile_model(model, loss='mse', learning_rate=1E-3, distributed=False):
    
    if(distributed==True):
        optt  = Adam(learning_rate=learning_rate * hvd.size())
        optt  = hvd.DistributedOptimizer(optt)
    else:
        optt  = Adam(learning_rate=learning_rate)  

    model.compile(loss=loss, optimizer=optt)

    return model


#Function to create nirb
def create_nirb_resnet(reduced_inputs, reduced_outputs, 
                       neurons_in_layer1=5,
                       neurons_in_layer2=64, 
                       learning_rate=1E-3,
                       distributed=False,
                       loss_function='mse',
                       info=False):

    #Set horovod
    if(distributed==True):
        hvd.init()

        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tensorflow.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    #Define input layer
    input_layer  = Input(shape=reduced_inputs.shape[1])
    input_layer  = Dense(reduced_inputs.shape[1], use_bias = False)(input_layer)
    input_layer  = BiasLayer()(input_layer)

    #Define structures
    z = input_layer
    z = keras.layers.Add()([resnet_block(z, neurons_in_layer1), z])
    z = keras.layers.Add()([resnet_block(z, neurons_in_layer2), z])
    
    #Define output layer
    output_layer = Dense(reduced_outputs.shape[1])(z)
    
    #Define model
    model = keras.models.Model(input_layer, output_layer)

    #Compile model
    model = compile_model(model, loss=loss_function, learning_rate=learning_rate, distributed=distributed)

    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())

    #Return model
    return model