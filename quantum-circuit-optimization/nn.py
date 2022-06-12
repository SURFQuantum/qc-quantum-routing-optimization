import numpy as np
from keras.layers import Dense, Reshape, Softmax
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential, save_model, load_model
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
import os
from save_data import load_object

directory= 'train_data/'
filepath = './saved_model'

y_train = []
state = []

for filename in sorted(os.listdir(directory)):

    if 'action.pickle_' in filename:
        one_hot = [[[1, 0], [1, 0], [1, 0], [1, 0]]]
        encoded = load_object(directory+filename)
        if encoded[0] == 0 or encoded[1] == 0:
            one_hot[0][0] = [0, 1]
        if encoded[0] == 1 or encoded[1] == 1:
            one_hot[0][1] = [0, 1]
        if encoded[0] == 2 or encoded[1] == 2:
            one_hot[0][2] = [0, 1]
        if encoded[0] == 3 or encoded[1] == 3:
            one_hot[0][3] = [0, 1]
        y_train.append(one_hot)

    elif 'state.pickle_' in filename:
        state.append((load_object(directory + filename).tolist()))

y_train = np.array(y_train)
state = np.array(state)

# Test Data
test = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# # Load the model
# model = load_model(filepath, compile = True)

model = Sequential()

model.add(Dense(10, input_dim=40, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Reshape((1,4,2), input_shape=(8,)))
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=adam_v2.Adam(learning_rate=0.01))

history = model.fit(state,y_train, verbose=2,epochs=110)
print(history.history.keys())

layer = Softmax()
prediction = np.round(layer(model.predict(test)).numpy())
print(prediction)

# # Save to model after it is trained
save_model(model, filepath)

plt.plot(history.history['loss'])
plt.title('model loss on 400 simulations')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()



