import tensorflow as tf
import numpy as np
from tensorflow import keras

EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
RESHAPE = 784

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, RESHAPE)
X_test = X_test.reshape(10000, RESHAPE)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train, X_test = X_train / 255.0, X_test / 255.0

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(RESHAPE,)))
model.add(keras.layers.Dense(
    N_HIDDEN,
    name='dense_layer',
    activation='relu'
))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(
    N_HIDDEN,
    name='dense_layer2',
    activation='relu'
))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(
    NB_CLASSES,
    name='dense_layer3',
    activation='softmax'
))

model.summary()

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