#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import keras
import dask.array as da
import horovod.tensorflow.keras as hvd
from keras import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import Multiply
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


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


#Function to perform reduction on output side, spatio temporal for large array
def pod_reduction_spatiotemporal_dask(snapshots, params_cut_off=99.999, memory_limit='2GB',
                                      save_path=None):
    
    #Check dimensions
    Ns = snapshots.shape[0]
    Nt = snapshots.shape[1]
    
    #Re-arrange
    r_snapshots = numpy.zeros((Nt*Ns, snapshots.shape[2]))
    
    aa = 0
    for ww in range(Ns):
        for kk in range(Nt):
            r_snapshots[aa,:] = snapshots[ww,kk,:]
            aa = aa + 1
    
    #Free memory
    snapshots = 0
    
    #Conversion to dask array
    r_snapshots = da.from_array(r_snapshots, chunks=memory_limit)
    
    #Perform SVD
    svd = da.linalg.svd_compressed(r_snapshots, k=Ns*Nt)
    
    #Compute eigenvalues only
    eig_vals = svd[1].compute()

    #Reduction
    basis_matrix = []
    basis_time_matrix = []
    information_content = []

    #Enumeration for included eigen vectors
    index = 0
    
    #Calculate total eigne values
    sum_eigen_full = numpy.sum(eig_vals**2)

    for j in range(len(eig_vals)):
        
        #Calculate time eigen sum
        sum_eigen = numpy.sum(eig_vals[:j]**2) 

        #Performing parmeters cut off
        information_content.append(sum_eigen/sum_eigen_full)
        if((sum_eigen/sum_eigen_full)<(params_cut_off/100)):
            index+=1
    
    basis_matrix = numpy.copy(svd[2].compute()[:index])
    
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


#Function to perform reduction on output side, temporal for large array
def pod_reduction_temporal_dask(snapshots, params_cut_off=99.999, memory_limit='2GB',
                                save_path=None):
    
    #Check dimensions
    Ns = snapshots.shape[0]
    
    #Conversion to dask array
    r_snapshots = da.from_array(snapshots, chunks=memory_limit)

    #Free memory
    snapshots = 0

    #Perform SVD
    svd = da.linalg.svd_compressed(r_snapshots, k=Ns)
    
    #Compute eigenvalues only
    eig_vals = svd[1].compute()

    #Reduction
    basis_matrix = []
    basis_time_matrix = []
    information_content = []

    #Enumeration for included eigen vectors
    index = 0
    
    #Calculate total eigne values
    sum_eigen_full = numpy.sum(eig_vals**2)

    for j in range(len(eig_vals)):
        
        #Calculate time eigen sum
        sum_eigen = numpy.sum(eig_vals[:j]**2) 

        #Performing parmeters cut off
        information_content.append(sum_eigen/sum_eigen_full)
        if((sum_eigen/sum_eigen_full)<(params_cut_off/100)):
            index+=1
    
    basis_matrix = numpy.copy(svd[2].compute()[:index])
    
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


#Function to create model
def bt_model(reduced_inputs_branch, reduced_inputs_trunk, reduced_outputs, 
             branch_neurons_in_layer1, branch_neurons_in_layer2, branch_activation_function,
             trunk_neurons_in_layer1, trunk_neurons_in_layer2, trunk_activation_function):

    branch_input_layer = Input(shape=(reduced_inputs_branch.shape[1],), name="branch_input")
    x = Dense(reduced_inputs_branch.shape[1])(branch_input_layer)
    if(branch_neurons_in_layer1!=0):
        x = Dense(branch_neurons_in_layer1, activation=branch_activation_function)(x)
    if(branch_neurons_in_layer2!=0):
        x = Dense(branch_neurons_in_layer2, activation=branch_activation_function)(x)
    branch_output_layer = Dense(reduced_outputs.shape[1])(x)
    
    trunk_input_layer = Input(shape=(1,), name="trunk_input")
    x = Dense(1)(trunk_input_layer)
    if(trunk_neurons_in_layer1!=0):
        x = Dense(trunk_neurons_in_layer1, activation=trunk_activation_function)(x)
    if(trunk_neurons_in_layer2!=0):
        x = Dense(trunk_neurons_in_layer2, activation=trunk_activation_function)(x)
    trunk_output_layer = Dense(reduced_outputs.shape[1])(x)

    output_layer = Multiply(name="output")([branch_output_layer, trunk_output_layer])
    
    m = Model([branch_input_layer, trunk_input_layer], output_layer)

    return m


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
def create_nirb_bt(reduced_inputs_branch, reduced_inputs_trunk, reduced_outputs, 
                   branch_neurons_in_layer1=5, branch_neurons_in_layer2=64, branch_activation_function='sigmoid',  
                   trunk_neurons_in_layer1=5, trunk_neurons_in_layer2=64, trunk_activation_function='sigmoid',
                   loss_function='mse', learning_rate=1E-3, distributed=False,
                   info=False):

    #Set horovod
    if(distributed==True):
        hvd.init()

        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tensorflow.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    #Define structures
    model = bt_model(reduced_inputs_branch, reduced_inputs_trunk, reduced_outputs, 
                     branch_neurons_in_layer1, branch_neurons_in_layer2, branch_activation_function,
                     trunk_neurons_in_layer1, trunk_neurons_in_layer2, trunk_activation_function)
 
    #Compile model
    model = compile_model(model, loss=loss_function, learning_rate=learning_rate, distributed=distributed)
    
    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())

    #Return model
    return model