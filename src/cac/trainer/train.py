#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import horovod.tensorflow.keras as hvd
import keras
from sklearn.metrics import log_loss, jaccard_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import CSVLogger
from keras.models import save_model
from keras.utils import set_random_seed
from PIL import Image
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


#Function to define training with affine transformation
def train_with_generator(training_images, training_labels, 
                         model, batch_size, epochs,
                         distributed=False,
                         save_path=None):
    
    #Import internal library
    from ..utilities import augmentation_generator
    
    #Initialize generator
    training_generator = augmentation_generator(training_images, training_labels, batch_size=batch_size, flip_axes=[1,2])
    steps_per_epoch_tr = len(numpy.array_split(numpy.zeros(len(training_images)), int(len(training_images)/batch_size)))
    
    #Training
    if(save_path!=None):

        if(distributed==True):
            #Record training
            csvlog = CSVLogger(save_path + '/training_log.csv')
            callbacks = [
                    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback(),
                    csvlog]

            #Fitting
            model.fit(training_generator,
                      steps_per_epoch=steps_per_epoch_tr // hvd.size(),
                      epochs=epochs,
                      verbose=0,
                      initial_epoch=0,
                      callbacks=callbacks)
        
            #Saving
            strr = save_path + "/trained_model.tf"
            save_model(model, strr)
        else:
            #Record training
            csvlog = CSVLogger(save_path + '/training_log.csv')

            #Fitting
            model.fit(training_generator,
                      steps_per_epoch=steps_per_epoch_tr,
                      epochs=epochs,
                      verbose=0,
                      initial_epoch=0,
                      callbacks=[csvlog])
        
            #Saving
            strr = save_path + "/trained_model.tf"
            save_model(model, strr)

    else:

        if(distributed==True):
            #Record training
            callbacks = [
                    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback()]

            #Fitting
            model.fit(training_generator,
                      steps_per_epoch=steps_per_epoch_tr // hvd.size(),
                      epochs=epochs,
                      verbose=0,
                      initial_epoch=0,
                      callbacks=callbacks)
        else:
            #Fitting
            model.fit(training_generator,
                      steps_per_epoch=steps_per_epoch_tr,
                      epochs=epochs,
                      verbose=0,
                      initial_epoch=0)

    return model


#Function to define training without affine transformation
#If you are confident with the quality of your set
def train_without_generator(training_images, training_labels, 
                            model, batch_size, epochs,
                            distributed=False,
                            save_path=None):
    
    #Initialize generator
    steps_per_epoch_tr = len(numpy.array_split(numpy.zeros(len(training_images)), int(len(training_images)/batch_size)))

    #Training
    if(save_path!=None):

        if(distributed==True):
            #Record training
            csvlog = CSVLogger(save_path + '/training_log.csv')
            callbacks = [
                    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback(),
                    csvlog]

            #Fitting
            model.fit(x=training_images, y=training_labels,
                    steps_per_epoch=steps_per_epoch_tr // hvd.size(),
                    epochs=epochs,
                    verbose=0,
                    callbacks=callbacks)
        
            #Saving
            strr = save_path + "/trained_model.tf"
            save_model(model, strr)
        else:
            #Record training
            csvlog = CSVLogger(save_path + '/training_log.csv')

            #Fitting
            model.fit(x=training_images, y=training_labels,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      callbacks=[csvlog])

            #Saving
            strr = save_path + "/trained_model.tf"
            save_model(model, strr)

    else:
        
        if(distributed==True):
            #Record training
            callbacks = [
                    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback()]

            #Fitting
            model.fit(x=training_images, y=training_labels,
                      steps_per_epoch=steps_per_epoch_tr // hvd.size(),
                      epochs=epochs,
                      verbose=0,
                      callbacks=callbacks)
        else:
            #Fitting
            model.fit(x=training_images, y=training_labels,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0)

    return model


#Function to calculate error prediction in image, only for binary
def calculate_image_metric(test_images, test_labels, 
                           model, 
                           save_path=None):

    #Using model to make prediction on test set
    p_test_labels = model.predict(test_images, batch_size=1)

    #Thresholding
    p_test_labels[p_test_labels < 0.5]  = 0
    p_test_labels[p_test_labels >= 0.5] = 1

    #Transfering to 2-D matrix
    true_matrix      = numpy.zeros((test_labels.shape[0], test_labels.shape[1] * test_labels.shape[2] * 1))
    predicted_matrix = numpy.zeros((p_test_labels.shape[0], p_test_labels.shape[1] * p_test_labels.shape[2] * 1))
    for ww in range(test_labels.shape[0]):
        true_matrix[ww, :] = test_labels[ww, :, :, :].flatten()
        predicted_matrix[ww, :] = p_test_labels[ww, :, :, :].flatten()
    
    #Error calculation
    test_error = log_loss(true_matrix, predicted_matrix)

    #Saving options
    if(save_path!=None):
        numpy.savetxt(save_path + '/test_error.txt', [test_error])

    #Return error
    return test_error