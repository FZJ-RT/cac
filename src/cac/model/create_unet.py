#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import horovod.tensorflow.keras as hvd
import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from keras.layers import concatenate 
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


#Function to create input layer
def input_layer(training_images):

    return Input(shape=training_images.shape[1:], name='input_layer')


#Function to create output layer
def output_layer(previous_layer, num_classes, activation):
    
    o = Conv2D(num_classes, (1, 1), padding='same')(previous_layer)
    o = Activation(activation,name='output')(o)

    return o


#Function to compile all architectures
def compile_model(input_layer, output_layer, loss_function, learning_rate=1E-3, distributed=False):
    
    model = Model(inputs=input_layer,outputs=output_layer)

    if(distributed==True):
        optt  = Adam(learning_rate=learning_rate * hvd.size())
        optt  = hvd.DistributedOptimizer(optt)
    else:
        optt  = Adam(learning_rate=learning_rate)  

    model.compile(loss={'output':loss_function}, optimizer=optt)

    return model


#Function to create encoder blocks
def encoder(previous_layer, pooling_size, filter_size, kernel_size, batchnorm_axis, is_first=False):
    
    if(is_first==False):
        e = MaxPooling2D(pooling_size)(previous_layer)
        e = Conv2D(filters=filter_size, kernel_size=kernel_size, padding='same')(e)
    else:
        e = Conv2D(filters=filter_size, kernel_size=kernel_size, padding='same')(previous_layer)
    e = BatchNormalization(axis=batchnorm_axis)(e)
    e = Activation('relu')(e)
    e = Conv2D(filters=filter_size, kernel_size=kernel_size, padding='same')(e)
    e = BatchNormalization(axis=batchnorm_axis)(e)
    e = Activation('relu')(e)

    return e


#Function to create decoder blocks
def decoder(previous_layer, coupled_layer, pooling_size, filter_size, kernel_size, batchnorm_axis):

    d = UpSampling2D(pooling_size,)(previous_layer)
    d = concatenate([coupled_layer,d], axis=-1)#skip connection
    d = Conv2DTranspose(filter_size, kernel_size, padding='same')(d)
    d = BatchNormalization(axis=batchnorm_axis)(d)
    d = Activation('relu')(d)
    d = Conv2DTranspose(filter_size, kernel_size, padding='same')(d)
    d = BatchNormalization(axis=batchnorm_axis)(d)
    d = Activation('relu')(d)

    return d


#Function to create unet
def create_vanilla_unet(training_images, training_labels,
                        num_classes=1, 
                        encoder_depth=5,
                        number_filters_at_first_encoder=64, 
                        pooling_size=(2,2), 
                        kernel_size=(3,3), 
                        batchnorm_axis=-1,
                        learning_rate=1E-3,
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

    #Define number of filters
    filter_size = numpy.zeros((encoder_depth,1))
    filter_size[0] = int(number_filters_at_first_encoder)
    for ww in range(1,encoder_depth):
        filter_size[ww] = int(2 * filter_size[ww-1])
    filter_size = filter_size.flatten()

    #Define input layer
    inputs = input_layer(training_images)
    
    #Define encoder 0
    globals()['e0'] = encoder(inputs, pooling_size, filter_size[0], kernel_size, batchnorm_axis, is_first=True)

    #Define other encoders
    for ww in range(1, encoder_depth):

        #Define encoder
        globals()['e' + str(ww)] = encoder(globals()['e' + str(ww-1)], pooling_size, filter_size[ww], kernel_size, batchnorm_axis)

    #Define decoder_<encoder_depth-2>
    globals()['d' + str(encoder_depth-2)] = decoder(globals()['e' + str(encoder_depth-1)], globals()['e' + str(encoder_depth-2)], pooling_size, filter_size[int(encoder_depth-2)], kernel_size, batchnorm_axis)

    #Define other decoders
    for ww in range(encoder_depth-3, -1, -1):

        #Define decoder
        globals()['d' + str(ww)] = decoder(globals()['d' + str(ww+1)], globals()['e' + str(ww)], pooling_size, filter_size[ww], kernel_size, batchnorm_axis)

    #Define output layer
    if(num_classes <= 2):
        outputs = output_layer(d0, num_classes=1, activation='sigmoid')
        model = compile_model(inputs, outputs, loss_function='binary_crossentropy', learning_rate=learning_rate, distributed=distributed)
    else:
        outputs = output_layer(d0, num_classes=num_classes, activation='softmax')
        model = compile_model(inputs, outputs, loss_function='categorical_crossentropy', learning_rate=learning_rate, distributed=distributed)
    
    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())
        
        s1in = training_images.shape[1]
        s2in = training_images.shape[2]
        for ww in range(encoder_depth):
            s1out = int( s1in/2 )
            s2out = int( s2in/2 )
            s1in  = s1out
            s2in  = s2out
        
        print("Your latent space has the size: (" + str(int(s1out)) + "," + str(int(s2out)) + ")")

    #Return model
    return model


#Function to load existing unet 
def load_existing_unet(kerasname=None, info=False):

    #Load the whole model in .keras file format
    model = load_model(kerasname, compile=False)

    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())
    
    #Return model
    return model