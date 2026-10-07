import tensorflow as tf
from tensorflow import keras

NB_CLASSES = 10
RESHAPE = 784

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(
    NB_CLASSES,
    input_shape=(RESHAPE,),
    kernel_initializer='zeros',
    name='dense_layer',
    activation='softmax'
))
