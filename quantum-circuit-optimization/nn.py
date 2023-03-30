import numpy as np
import matplotlib.pyplot as plt
import os
from save_data import load_object
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Softmax
import torch

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

y_train = torch.Tensor(y_train)
state = np.array(state)

# Test Data
test = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# # Load the model
# model = load_model(filepath, compile = True)

class NNmodel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = Sequential(
                Linear(40, 10),
                ReLU(),
                Linear(10, 10),
                ReLU(),
                Linear(10, 10),
                ReLU(),
                Linear(10, 8),
                Softmax()
        )

    def forward(self, x):
        return self.model(x)


model = NNmodel()

test = torch.from_numpy(test).to(dtype=torch.float32)
prediction = model(test).round()
print(prediction)

# # Save to model after it is trained
torch.save(model, filepath)
