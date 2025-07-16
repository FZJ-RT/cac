#-----------------------------------------------------
#Import external libraries
import numpy
import keras
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
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


#Function to create gpr model
def gpr(length_scale, nu):

    return Matern(length_scale=length_scale, 
                  length_scale_bounds="fixed",
                  nu=nu)


#Function to compile all architectures
def compile_model(model, inputs, outputs):

    return GaussianProcessRegressor(kernel=model,
                                    random_state=0,
                                    n_restarts_optimizer=0).fit(
                                    inputs, outputs)


#Function to create nirb
def create_nirb_gpr(reduced_inputs, reduced_outputs, 
                    length_scale,
                    nu):

    #Define structures
    model = gpr(length_scale, nu)
 
    #Compile model
    model = compile_model(model, reduced_inputs, reduced_outputs)
    
    #Return model
    return model