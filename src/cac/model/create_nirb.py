#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import keras
import dask.array as da
import horovod.tensorflow.keras as hvd
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


#Function to perform reduction on output side, spatio temporal
#Check Degen et al. (2023) - Perspectives of physics-based machine learning strategies for geoscientific applications governed by partial differential equations
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
#Check Degen et al. (2023) - Perspectives of physics-based machine learning strategies for geoscientific applications governed by partial differential equations
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


#Function to create regressional blocks
def neural_network(reduced_inputs, reduced_outputs, neurons_in_layer1, neurons_in_layer2, activation_function):

    m = Sequential()
    m.add(Dense(reduced_inputs.shape[1]))
    if(neurons_in_layer1!=0):
        m.add(Dense(neurons_in_layer1, activation=activation_function))
    if(neurons_in_layer2!=0):
        m.add(Dense(neurons_in_layer2, activation=activation_function))        
    m.add(Dense(reduced_outputs.shape[1]))

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
def create_nirb(reduced_inputs, reduced_outputs, 
                neurons_in_layer1=5,
                neurons_in_layer2=64, 
                learning_rate=1E-3,
                activation_function='sigmoid',
                loss_function='mse',
                distributed=False,
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
    model = neural_network(reduced_inputs, reduced_outputs, neurons_in_layer1, neurons_in_layer2, activation_function)
 
    #Compile model
    model = compile_model(model, loss=loss_function, learning_rate=learning_rate, distributed=distributed)
    
    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())

    #Return model
    return model