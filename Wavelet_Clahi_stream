import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Layer

# Define the ChannelAttention Layer
class ChannelAttention(Layer):
    def _init_(self, d_model, ratio, **kwargs):
        super(ChannelAttention, self)._init_(**kwargs)
        self.d_model = d_model
        self.ratio = ratio
        
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units=d_model//ratio, activation='relu')
        self.dense2 = Dense(units=d_model, activation='sigmoid')
    
    def build(self, input_shape):
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        dense1 = self.dense1(avg_pool)
        dense2 = self.dense2(dense1)
        dense2 = tf.reshape(dense2, [-1, 1, 1, self.d_model])
        return inputs * dense2

# Define input shapes for both streams
# input_shape_wavelet = (64, 64, 1)
# input_shape_clahe = (64, 64, 1)

# Wavelet Stream
#input_tensor_wavelet = Input(shape=input_shape_wavelet)
x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_tensor_wavelet)
x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D((2, 2), strides=2)(x1)
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D((2, 2), strides=2)(x1)
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
x1 = Dropout(0.3)(x1)
x1 = BatchNormalization()(x1)

# CLAHE Stream
#input_tensor_clahe = Input(shape=input_shape_clahe)
x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_tensor_wavelet)
x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(x2)
x2 = MaxPooling2D((2, 2), strides=2)(x2)
x2 = Conv2D(128, (5, 5), activation='relu', padding='same')(x2)
x2 = Conv2D(128, (5, 5), activation='relu', padding='same')(x2)
x2 = MaxPooling2D((2, 2), strides=2)(x2)
x2 = Conv2D(512, (5, 5), activation='relu', padding='same')(x2)
x2 = Dropout(0.3)(x2)
x2 = BatchNormalization()(x2)

# print("Wavelet stream output shape:", x1.shape)
# print("CLAHE stream output shape:", x2.shape)

# Concatenate the outputs of the two streams
concatenated = Concatenate()([x1, x2])

# Apply Channel Attention after concatenation and before flattening
attention_output = ChannelAttention(d_model=concatenated.shape[-1], ratio=8)(concatenated)

# Flatten and fully connected layers
flattened = Flatten()(attention_output)
fc1 = Dense(128, activation='relu')(flattened)
output = Dense(3, activation='softmax')(fc1)

# Create the model
model = Model(inputs=[input_tensor_wavelet, input_tensor_wavelet], outputs=output)

# Print model summary
model.summary()
