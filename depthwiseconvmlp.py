import tensorflow as tf
from tensorflow.keras import layers, models

def depthwise_conv_mlp(input_shape, depth_multiplier, pointwise_filters, hidden_units, output_dim, activation='relu'):
    """
    Creates a model with Depthwise Convolution followed by Point-wise Convolution and a small MLP.
    
    Args:
    - input_shape (tuple): The shape of the input, e.g., (H, W, C) for images.
    - depth_multiplier (int): Multiplier for the depthwise convolution, determines the number of output channels per input channel.
    - pointwise_filters (int): Number of filters in the 1x1 point-wise convolution.
    - hidden_units (list of int): List where each element represents the number of neurons in that hidden layer.
    - output_dim (int): The dimension of the output features.
    - activation (str): Activation function to use in hidden layers.
    
    Returns:
    - model (tf.keras.Model): The combined Depthwise Convolution, Point-wise Convolution, and MLP model.
    """
    inputs = layers.Input(shape=input_shape)

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), depth_multiplier=depth_multiplier, activation=activation, padding='same')(inputs)
    
    # Point-wise Convolution (1x1 convolution)
    x = layers.Conv2D(filters=pointwise_filters, kernel_size=(1, 1), activation=activation)(x)
    
    # Flatten the output from the convolution to feed into the MLP
    x = layers.Flatten()(x)
    
    # Small MLP
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
    
    # Output layer
    outputs = layers.Dense(output_dim)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
input_shape = (64, 64, 256)    # Input of size 64x64 with 256 channels
depth_multiplier = 1           # Depth multiplier for the depthwise convolution
pointwise_filters = 128        # Reduce the channels to 128 with the 1x1 point-wise convolution
hidden_units = [64, 32]        # Two hidden layers in the MLP with 64 and 32 units
output_dim = 10                # Output dimension, e.g., for classification (10 classes)

small_model = depthwise_conv_mlp(input_shape, depth_multiplier, pointwise_filters, hidden_units, output_dim)

# Display the model architecture
small_model.summary()
