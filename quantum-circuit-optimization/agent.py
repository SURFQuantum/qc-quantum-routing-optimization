import numpy as np
import random
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.adamax import Adamax
from keras.optimizers import adam_v2
from keras.models import model_from_json

from environment import Environment
from circuit import Circuit


# class state defines the board and decides reward, end and next position
class State:
    def __init__(self, circuit, agent):
        self.circuit = circuit.get_circuit()  # [[0, 1, 0], [3, 2, 0], [3, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0], [2, 3, 0]]
        self.isEnd = False
        self.length = 0
        self.scheduled_gates = agent.scheduled_gates
        self.action = agent.action()

    # TODO: define new reward system
    def get_reward(self):
        return 0

    def circuit_length(self, circuit):
        self.length = len(circuit)

    # TODO: check this function
    def is_end_circuit(self, circuit_i):
        if circuit_i == self.length - 1:
            self.isEnd = True

    # TODO: write new action
    def next_position(self, action):
        return 0

    def state(self, circuit_i):
        location_qubit = circuit_i[0]
        interaction_qubit = circuit_i[1]
        scheduled_gates = self.scheduled_gates
        #TODO: find proper way to calculate distance based on topology
        distance = circuit_i[1] - circuit_i[0]

        state = [location_qubit, interaction_qubit, scheduled_gates, distance]
        return state


class Agent:

    def __init__(self, environment):
        # inialise states and actions
        self.allowSchedule = environment.circuit_connectivity_compare()  # Boolean
        self.scheduled_gates = []

    def model(self):

        # TODO: create vector where length = input_size
        input_size = 3
        model = Sequential()
        model.add(Dense(10, input_dim=input_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # TODO: try also different optimizers
        return model

    # method to choose action with Epsilon greedy policy, and move to next state
    def action(self):
        return

    def schedule_gate(self,gate):
        if self.allowSchedule(gate):
            self.scheduled_gates.append(gate)

    #TODO: print out tree of MCTS
    def show_tree(self):
        return


if __name__ == "__main__":
    # create agent for 10,000 episdoes implementing a Q-learning algorithm plot and show values.
    ag = Agent()

