#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import horovod.tensorflow.keras as hvd
import keras
from keras.models import Model, Sequential
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#-----------------------------------------------------


#Function to create input layer for encoder
def encoder_input_layer(training_images):

    return Input(shape=training_images.shape[1:])


#Function to create convolutional layer for encoder
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


#Function to create latent layer for encoder
def encoder_latent(previous_layer, neurons, latent_dim):
    
    l = Flatten()(previous_layer)
    l = Dense(neurons, activation="relu")(l)
    l = Dense(latent_dim)(l)

    return l


#Function to create input layer for decoder
def decoder_input_layer(latent_dim):

    return Input(shape=(latent_dim,))


#Function to create latent layer for decoder
def decoder_latent(previous_layer, neurons, last_filter):
    
    d = Dense(neurons, activation="relu")(previous_layer)
    d = Dense(int(last_filter * neurons * neurons), activation='relu')(d)
    d = Reshape((neurons, neurons, int(last_filter)))(d)

    return d

#Function to create convolutional layer for decoder
def decoder(previous_layer, pooling_size, filter_size, kernel_size, batchnorm_axis, is_first=False):
    
    if(is_first==False):
        d = UpSampling2D(pooling_size,)(previous_layer)
        d = Conv2DTranspose(filter_size, kernel_size, padding='same')(d)
    else:
        d = Conv2DTranspose(filter_size, kernel_size, padding='same')(previous_layer)
    d = BatchNormalization(axis=batchnorm_axis)(d)
    d = Activation('relu')(d)
    d = Conv2DTranspose(filter_size, kernel_size, padding='same')(d)
    d = BatchNormalization(axis=batchnorm_axis)(d)
    d = Activation('relu')(d)

    return d


#Function to create output layer for decoder
def decoder_output_layer(previous_layer, kernel_size):

    return Conv2DTranspose(filters=1, kernel_size=kernel_size, activation="sigmoid", padding='same')(previous_layer)


#Function to create cnn
def create_cae2D(training_labels,
                 latent_dim=2, 
                 encoder_depth=2, 
                 number_filters_at_first_encoder=32, 
                 number_last_neurons=16,
                 pooling_size=(2,2), 
                 stride_size=(2,2),
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

    #Define encoder
    encoder_inputs = encoder_input_layer(training_labels)

    globals()['e0'] = encoder(encoder_inputs, pooling_size, filter_size[0], kernel_size, batchnorm_axis, is_first=True)
    for ww in range(1, encoder_depth):
        globals()['e' + str(ww)] = encoder(globals()['e' + str(ww-1)], pooling_size, filter_size[ww], kernel_size, batchnorm_axis)
    
    globals()['l0'] = encoder_latent(globals()['e' + str(encoder_depth-1)], number_last_neurons, latent_dim)
    encoding = Model(encoder_inputs, l0, name="encoding")
    

    #Define decoder
    decoder_inputs = decoder_input_layer(latent_dim)
    decoder_lt     = decoder_latent(decoder_inputs, number_last_neurons, filter_size[int(encoder_depth-1)])
    
    globals()['d' + str(encoder_depth-1)] = decoder(decoder_lt, pooling_size, filter_size[ww], kernel_size, batchnorm_axis, is_first=True)
    for ww in range(encoder_depth-2, -1, -1):
        globals()['d' + str(ww)] = decoder(globals()['d' + str(ww+1)], pooling_size, filter_size[ww], kernel_size, batchnorm_axis)

    decoder_outputs = decoder_output_layer(d0, kernel_size[0])
    decoding = Model(decoder_inputs, decoder_outputs, name="decoding")
    
    class ae_compile(Model):
        def __init__(self, encoder, decoder):
            super(ae_compile, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    model = ae_compile(encoding, decoding)

    if(distributed==True):
        optt  = Adam(learning_rate=learning_rate * hvd.size())
        optt  = hvd.DistributedOptimizer(optt)
    else:
        optt  = Adam(learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=optt)
    
    #Display
    if(info==True):
        print("This is your model's encoder:")
        print(encoding.summary())

        print("\n\nThis is your model's decoder:")
        print(decoding.summary())

    #Return model
    return model


#Function to load existing autoencoder
def load_existing_cae2D(encoder_kerasname=None, 
                        decoder_kerasname=None,
                        info=False):

    #Load the whole model in .keras file format
    encoder = load_model(encoder_kerasname, compile=False)
    decoder = load_model(decoder_kerasname, compile=False)

    #Display
    if(info==True):
        print("This is your model's encoder:")
        print(encoder.summary())

        print("\n\nThis is your model's decoder:")
        print(decoder.summary())
    
    #Return model
    return encoder, decoder