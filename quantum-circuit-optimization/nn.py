import numpy as np
from keras import Input
from keras.layers import Embedding, Dense, Reshape, Softmax
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import adam_v2
from matplotlib import pyplot as plt

input_size = 28
y_pred = [[0.1, 0.9],[0.4, 0.6], [0.3, 0.7], [0.99, 0.01]]

y_true = np.array([[[0,1], [0,1], [0,1], [1, 0]]])

state = np.array([[1, 1, 1, 1, 1,  0,  0, 1, 1,  0, 1,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

model = Sequential()

model.add(Dense(10, input_dim=input_size, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Reshape((4, 2), input_shape=(8,)))
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=adam_v2.Adam(lr=0.1))

history = model.fit(state,y_true, verbose=2,epochs=5)

layer = Softmax()
print(layer(model.predict(state)).numpy())

