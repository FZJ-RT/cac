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
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose, Reshape
from keras.layers import concatenate 
from keras.layers import Multiply, EinsumDense
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility
#-----------------------------------------------------


#Function to create input layer for image
def input_layer_image(training_images):
    
    return Input(shape=training_images.shape[1:], name='input_layer_image')

#Function to create input layer for scalar
def input_layer_scalar(modification_value):

    return Input(shape=(1,), name='input_layer_scalar')


#Function to create output layer --changes
def output_layer(previous_layer, unet_output_activation_function='sigmoid'):
    
    o = Conv2D(1, (1, 1), padding='same')(previous_layer)
    o = Activation(unet_output_activation_function,name='output')(o)

    return o


#Function to compile all architectures
def compile_model(input_layer_image, input_layer_scalar, output_layer, learning_rate=1E-3, distributed=False):
    
    model = Model([input_layer_image, input_layer_scalar], output_layer)

    if(distributed==True):
        optt  = Adam(learning_rate=learning_rate * hvd.size())
        optt  = hvd.DistributedOptimizer(optt)
    else:
        optt  = Adam(learning_rate=learning_rate)  
    
    def my_loss_fn(y_true, y_pred):
        return  tensorflow.reduce_mean(tensorflow.square(y_true-y_pred))/tensorflow.reduce_mean(tensorflow.square(y_true))

    model.compile(loss=my_loss_fn, optimizer=optt)

    return model


#Function to create encoder blocks
def encoder(previous_layer, pooling_size, filter_size, kernel_size, batchnorm_axis, unet_activation_function='relu', is_first=False):
    
    if(is_first==False):
        e = MaxPooling2D(pooling_size)(previous_layer)
        e = Conv2D(filters=filter_size, kernel_size=kernel_size, padding='same')(e)
    else:
        e = Conv2D(filters=filter_size, kernel_size=kernel_size, padding='same')(previous_layer)
    e = BatchNormalization(axis=batchnorm_axis)(e)
    e = Activation(unet_activation_function)(e)
    e = Conv2D(filters=filter_size, kernel_size=kernel_size, padding='same')(e)
    e = BatchNormalization(axis=batchnorm_axis)(e)
    e = Activation(unet_activation_function)(e)

    return e


#Function to create decoder blocks
def decoder(previous_layer, coupled_layer, conditioning_layer, pooling_size, filter_size, kernel_size, batchnorm_axis, unet_activation_function='relu'):
    
    #Main decoder
    d = UpSampling2D(pooling_size,)(previous_layer)
    
    #Conditioning
    c = Dense(d.shape[-1], activation=None)(conditioning_layer)
    e = Multiply()([d,c]) #we can use einsum if the conditioning input is 2D ijkl, pl -> ipjkl to add more snapshots
    
    d = concatenate([coupled_layer,e], axis=-1)#skip connection
    d = Conv2DTranspose(filter_size, kernel_size, padding='same')(d)
    d = BatchNormalization(axis=batchnorm_axis)(d)
    d = Activation(unet_activation_function)(d)
    d = Conv2DTranspose(filter_size, kernel_size, padding='same')(d)
    d = BatchNormalization(axis=batchnorm_axis)(d)
    d = Activation(unet_activation_function)(d)

    return d


#Function to modify latent space
def conditioning(input_layer, neurons_in_layer1, neurons_in_layer2, conditioning_activation_function='sigmoid'):

    c = Dense(input_layer.shape[1])(input_layer)
    if(neurons_in_layer1!=0):
        c = Dense(neurons_in_layer1, activation=conditioning_activation_function)(c)
    if(neurons_in_layer2!=0):
        c = Dense(neurons_in_layer2, activation=conditioning_activation_function)(c)

    return c


#Function to create unet --changes
def create_tfunet(training_images, modification_value, training_labels, 
                  encoder_depth=5,
                  number_filters_at_first_encoder=64, 
                  neurons_in_layer1=128,
                  neurons_in_layer2=128,
                  pooling_size=(2,2), 
                  kernel_size=(3,3), 
                  batchnorm_axis=-1,
                  learning_rate=1E-3,
                  distributed=False,
                  unet_activation_function='relu',
                  unet_output_activation_function='sigmoid',
                  conditioning_activation_function='sigmoid',
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

    #Define input layer for image
    input_image = input_layer_image(training_images)

    #Define input layer for scalar
    input_scalar = input_layer_scalar(modification_value)
    
    #Define conditioning
    cond = conditioning(input_scalar, neurons_in_layer1, neurons_in_layer2, conditioning_activation_function=conditioning_activation_function)

    #Define encoder 0
    globals()['e0'] = encoder(input_image, pooling_size, filter_size[0], kernel_size, batchnorm_axis, unet_activation_function=unet_activation_function, is_first=True)

    #Define other encoders
    for ww in range(1, encoder_depth):

        #Define encoder
        globals()['e' + str(ww)] = encoder(globals()['e' + str(ww-1)], pooling_size, filter_size[ww], kernel_size, batchnorm_axis, unet_activation_function=unet_activation_function)
    
    #Define decoder_<encoder_depth-2>
    globals()['d' + str(encoder_depth-2)] = decoder(globals()['e' + str(encoder_depth-1)], globals()['e' + str(encoder_depth-2)], cond, pooling_size, filter_size[int(encoder_depth-2)], kernel_size, batchnorm_axis, unet_activation_function=unet_activation_function)

    #Define other decoders
    for ww in range(encoder_depth-3, -1, -1):

        #Define decoder
        globals()['d' + str(ww)] = decoder(globals()['d' + str(ww+1)], globals()['e' + str(ww)], cond, pooling_size, filter_size[ww], kernel_size, batchnorm_axis, unet_activation_function=unet_activation_function)

    #Define output layer
    outputs = output_layer(d0)

    #Compile model
    model = compile_model(input_image, input_scalar, outputs, learning_rate=learning_rate, distributed=distributed)
   
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


#Function to load existing tfunet 
def load_existing_tfunet(kerasname=None, info=False):

    #Load the whole model in .keras file format
    model = load_model(kerasname, compile=False)

    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())
    
    #Return model
    return model