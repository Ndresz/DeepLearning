import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.layers import Dense

# Generate some dummy binary classification data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = K.Sequential([
    Dense(1, input_shape=(2,), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

loss, acc = model.evaluate(X, y)
print("Test accuracy:", acc)