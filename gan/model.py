### References ### 
# [1] Ledig, C.; Theis, L.; Huszar, F.; Caballero, J.; Aitken, A. P.; Tejani, A.; Totz, J.; Wang, Z., Shi, W., 
#     "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," CoRR, 2016.
#     https://arxiv.org/pdf/1609.04802.pdf
# [2] https://stackoverflow.com/questions/47862262/how-to-subtract-channel-wise-mean-in-keras/47862836#47862836
# [3] https://datascience.stackexchange.com/questions/56377/how-and-why-to-rescale-image-range-between-0-1-and-1-1

import os
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as Layers
from tensorflow.python.keras.layers import add as ElementwiseSum
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg19 import VGG19
from gan.utils import get_shape

### Model hyperparameters ###
# IO:
INPUT_SIZE = 64
TARGET_SIZE = 128
DATA_FORMAT = 'channels_last'

# Number of blocks
NUM_RESIDUAL_BLOCKS = 16
NUM_UPSAMPLING_BLOCKS = 1
NUM_CONV_BLOCKS = 5

# Convolutional parameters
KERNEL_SIZE = (3,3)
KERNEL_SIZE_CONV_STEP = (4,4)
KERNEL_SIZE_IO = (9,9)
STRIDES=(1,1)
STRIDES_CONV_STEP=(2,2)
PADDING='same'

# Batch normalization parameters
AXIS = -1

# Activation function parameters
ALPHA = 0.2
ALPHA_INITIALIZER='zeros'
SHARED_AXES = [1,2]

# Optimizer parameters
LEARNING_RATE = 1e-4
BETA = 0.9

# Activation functions
GENERATOR_ACTIVATION = 'tanh'
DISCRIMINATOR_ACTIVATION = 'sigmoid'

# LOSS
CXENT_LOSS = 'binary_crossentropy'
GAN_CONTENT_LOSS_WEIGHT = 1. 
GAN_PIXELWISE_LOSS_WEIGHT = 1e-3

def gan(gen, disc, channels='channels_last'):
    # Update image data format
    update_image_data_format(channels)

    # Create GAN input layer
    inputs = Layers.Input(shape=get_shape(INPUT_SIZE, DATA_FORMAT), name='input_gan')
    
    # Create generator and discriminator output layers
    outputs_generator = gen(inputs)
    outputs_discriminator = disc(outputs_generator)

    # Create a GAN model
    gan = Model(inputs=inputs, outputs=[outputs_generator, outputs_discriminator])

    # Make discriminator untrainable during the gan training process
    disc.trainable = False

    # Combine model loss functions
    loss = [content_loss, CXENT_LOSS]
    loss_weights=[GAN_CONTENT_LOSS_WEIGHT, GAN_PIXELWISE_LOSS_WEIGHT]

    # Create Optimizer
    optimizer=Adam(lr=LEARNING_RATE, beta_1=BETA)

    # Compile the model
    gan.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    return gan

def generator(channels='channels_last'):   
    # Update image data format
    update_image_data_format(channels)

    # Create input layer   
    inputs = Layers.Input(shape=get_shape(None, DATA_FORMAT), name='input_generator')

    # Add initial blocks
    x = Layers.Conv2D(filters=INPUT_SIZE, kernel_size=KERNEL_SIZE_IO, strides=STRIDES, padding=PADDING, activation=None)(inputs)
    x_skip_branch = Layers.PReLU(alpha_initializer=ALPHA_INITIALIZER,alpha_regularizer=None,alpha_constraint=None,shared_axes=SHARED_AXES)(x)
    x = x_skip_branch

    # Add the residual blocks
    for i in range(NUM_RESIDUAL_BLOCKS):
        res_block = Layers.Conv2D(filters=INPUT_SIZE, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, activation=None, use_bias=False)(x)
        res_block = Layers.BatchNormalization(axis=AXIS)(res_block)
        res_block = Layers.PReLU(alpha_initializer=ALPHA_INITIALIZER, alpha_regularizer=None, alpha_constraint=None, shared_axes=SHARED_AXES)(res_block)
        res_block = Layers.Conv2D(filters=INPUT_SIZE, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, activation=None, use_bias=False)(res_block)
        res_block = Layers.BatchNormalization(axis=AXIS)(res_block)
        x = ElementwiseSum([res_block, x])

    x = Layers.Conv2D(filters=INPUT_SIZE, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING, activation=None, use_bias=False)(x)
    x = Layers.BatchNormalization(axis=AXIS)(x)
    x = ElementwiseSum([x, x_skip_branch])
    
    # Add upsampling blocks
    for i in range(NUM_UPSAMPLING_BLOCKS):
        u = Layers.Conv2D(filters=INPUT_SIZE, kernel_size=KERNEL_SIZE, strides=STRIDES , padding=PADDING, activation=None, use_bias=False)(x)
        u = Layers.UpSampling2D(size=(2, 2))(u)
        x = Layers.PReLU(alpha_initializer=ALPHA_INITIALIZER, alpha_regularizer=None, alpha_constraint=None, shared_axes=SHARED_AXES)(u)

    # Create output layer
    outputs = Layers.Conv2D(filters=3, kernel_size=KERNEL_SIZE_IO, strides=STRIDES, padding=PADDING, activation=GENERATOR_ACTIVATION, use_bias=False)(x)

    # Create a generator model
    generator = Model(inputs=inputs, outputs=outputs, name='model_generator') 

    return generator

def discriminator(channels='channels_last'):
    def get_filter(x):
        return INPUT_SIZE * 2 ** int((x + 1) / 2)
        
    # Update image data format
    update_image_data_format(channels)

    # Create input layer
    inputs = Layers.Input(shape=get_shape(TARGET_SIZE, DATA_FORMAT), name='input_discriminator')

    # Add initial blocks
    x = Layers.Conv2D(filters=INPUT_SIZE, kernel_size=KERNEL_SIZE, strides=STRIDES, padding=PADDING,activation=None, use_bias=False)(inputs)
    x = Layers.LeakyReLU(alpha=ALPHA)(x)

    # Add convolutional blocks to the model
    for i in range(NUM_CONV_BLOCKS):
        # Calculate covnolutional step parameters
        filters = get_filter(i)
        kernel_size = KERNEL_SIZE if i % 2 > 0 else KERNEL_SIZE_CONV_STEP
        strides = STRIDES if i % 2 > 0 else STRIDES_CONV_STEP

        x = Layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=PADDING, activation=None, use_bias=False)(x)
        x = Layers.BatchNormalization(axis=AXIS)(x)
        x = Layers.LeakyReLU(alpha=ALPHA)(x)
    
    # Convert multi-dimensional signal into vector
    x = Layers.Flatten(data_format=DATA_FORMAT)(x)
    
    # Add fully connected layer
    x = Layers.Dense(units=get_filter(NUM_CONV_BLOCKS), activation=None)(x)
    x = Layers.LeakyReLU(alpha=ALPHA)(x)
    
    # Create output layer
    outputs = Layers.Dense(units=1, activation=DISCRIMINATOR_ACTIVATION)(x)

    # Create a discriminator model
    discriminator = Model(inputs, outputs, name='model_discriminator')
    
    # Create Optimizer
    optimizer = Adam(lr=LEARNING_RATE, beta_1=BETA)

    # The model needs to be compiled for a separate from GAN training
    discriminator.compile(loss=CXENT_LOSS, optimizer=optimizer)

    return discriminator

def update_image_data_format(channels):
    K.set_image_data_format(channels)
    DATA_FORMAT = channels
    AXIS = -1 if channels=='channels_last' else 1
    SHARED_AXES = [1,2] if channels=='channels_last' else [2,3]

def preproces_to_vgg(x):
    # Rescale  from [-1,1] to [0, 255]. Reverse the equation from [2] and [3].
    x += 1.
    x *= 127.5
    
    # Convert RGB to BGR
    x = x[..., ::-1] if DATA_FORMAT == 'channels_last' else x[:,::-1,:,:]

    # The mean BGR values of ImageNet training set.
    _IMAGENET_MEAN = K.constant(-np.array([103.939, 116.779, 123.68]))

    # Zero-center by mean pixel
    x = K.bias_add(x, K.cast(_IMAGENET_MEAN, K.dtype(x)))
    
    return x

def content_loss(hr, sr):
    # Load pre-trained 19 layer VGG network
    vgg19 = VGG19(include_top=False, input_shape=get_shape(TARGET_SIZE, DATA_FORMAT), weights='imagenet')

    # Make the loaded model untrainable
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False

    # The VGG loss is based on the ReLU activation layers of the pre-trained 19 layer VGG network
    # Extract the feature map obtained by the j-th convolution (after activation) before the i-th maxpooling layer
    # within the VGG19 network. According to research, the VGG54 gives the perceptually most convincing results [1].
    feature_map_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output)
    
    # compute the features, 'hr' and 'sr' are tensors scaled in [-1, 1]
    # it needs te be rescaled and shifted to respect VGG preprocessing strategy
    features_pred = feature_map_model(preproces_to_vgg(sr))
    features_true = feature_map_model(preproces_to_vgg(hr))
    
    # Make VGG losses of a scale that is comparable to the MSE loss. This is equivalent to multiplying with a rescaling factor of ≈ 0.006 [1].
    rescalig_factor = 0.006

    # Calculate according to Equation 5 in [1].
    # The Content loss, a combination of VGG and MSE loss, is the euclidean distance between the feature 
    # representations of a reconstructed image from generator and the reference high-resolution image.
    content_loss = rescalig_factor*K.mean(K.square(features_pred - features_true), axis=-1)   
    
    return content_loss

if __name__ == '__main__':
    generator = generator()
    generator.summary()

    discriminator = discriminator()
    discriminator.summary()

    gan = gan(generator, discriminator)
    gan.summary()
