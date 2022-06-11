import pickle
from os.path import exists
import numpy as np

def save_action(obj):

    count = 0
    try:
        while exists(f"train_data/action.pickle_{count}"):
            count+=1
        with open(f"train_data/action.pickle_{count}", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def save_state(obj):

    count = 0
    try:
        while exists(f"train_data/state.pickle_{count}"):
            count+=1
        with open(f"train_data/state.pickle_{count}", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


'''
Test Data
'''
# state = np.array([[1,1,1,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                    [1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                    [1,1,1,1,1,0,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                    [1,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                    [1,1,1,1,1,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                    [1,1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
#
# y_true = np.array([[[[0,1],[0,1],[1,0],[1,0]]], [[[0,1],[1,0],[0,1],[1,0]]], [[[0,1],[1,0],[1,0],[0,1]]],
#                   [[[1,0],[0,1],[0,1],[1,0]]], [
# state = np.array([1,1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
# #
# action = np.array([[[1,0],[0,1],[1,0],[0,1]]])
# save_state(state)[[1,0],[1,0],[0,1],[0,1]]], [[[1,0],[0,1],[1,0],[0,1]]]])
# save_action(action)
