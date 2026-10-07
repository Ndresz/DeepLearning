import tensorflow as tf
from tensorflow import keras
import numpy as np

EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10         
VALIDATION_SPLIT = 0.2   

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

RESHAPE = 784
X_train = X_train.reshape(60000, RESHAPE)
X_test = X_test.reshape(10000, RESHAPE)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255
X_test = X_test / 255

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(RESHAPE,)))
model.add(keras.layers.Dense(
    NB_CLASSES,
    name='dense_layer',
    activation='softmax'
))

model.compile(
    optimizer='SGD',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)