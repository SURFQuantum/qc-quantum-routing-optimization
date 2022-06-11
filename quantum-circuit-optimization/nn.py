import numpy as np
from keras.layers import Dense, Reshape, Softmax
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential, save_model, load_model
from keras.optimizers import adam_v2
import os
from save_data import load_object

directory= 'train_data/'
filepath = './saved_model'

y_train = []
state = []
count = 0
for filename in sorted(os.listdir(directory)):

    if filename == f'action.pickle_{count}':
        print(filename)
        y_train.append(load_object(directory+filename))
        count += 1

count = 0
for filename in sorted(os.listdir(directory)):
    if filename == f'state.pickle_{count}':
        print(filename)
        state.append(load_object(directory + filename))
        count+=1

y_train = np.array(y_train)
print(y_train)
state = np.array(state)
print(state)

# Test Data
test = np.array([[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# # Load the model
# model = load_model(filepath, compile = True)

model = Sequential()

model.add(Dense(10, input_dim=28, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Reshape((1,4,2), input_shape=(8,)))
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=adam_v2.Adam(learning_rate=0.01))

model.fit(state,y_train, verbose=2,epochs=100)

layer = Softmax()
prediction = np.round(layer(model.predict(test)).numpy())
print(prediction)

# # Save to model after it is trained
save_model(model, filepath)



