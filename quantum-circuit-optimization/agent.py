import keras.losses
from keras.optimizer_v2.adam import Adam

import environment
from monte_carlo_ts import MCTS
from environment import Environment
from circuit import Circuit

# class state defines the board and decides reward, end and next position
from qubit_allocation import Allocation
import config

from keras import Input, Model
from keras.layers.core import Dense
from keras.optimizer_v2 import rmsprop
from keras.regularizers import l2


class State:
    def __init__(self, circuit, agent):
        self.circuit = circuit.get_circuit()  # [[0, 1, 0], [3, 2, 0], [3, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0], [2, 3, 0]]
        self.isEnd = False
        self.length = 0
        self.scheduled_gates = agent.scheduled_gates
        self.action = agent.action()

    # TODO: get reward from MCTS
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
        # TODO: find proper way to calculate distance based on topology
        distance = circuit_i[1] - circuit_i[0]

        state = [location_qubit, interaction_qubit, scheduled_gates, distance]
        return state


class Agent:

    def __init__(self, circuit):
        self.learning_rate = 0.1
        self.scheduled_gates = []
        self.model = None
        self.n_qubits = circuit.n_qubits



    def build_model(self):

        # Input is number of qubits * maximum number swap gates that can be scheduled to decide the dimension of the circuit + the timestep entry
        input_size = (self.n_qubits*(self.n_qubits-2)+1)
        inputs = Input(shape=(input_size,))
        dense = Dense(8, activation="relu")
        x = dense(inputs)
        x = Dense(8, activation="relu")(x)
        outputs = Dense(10)(x)

        model = Model(inputs=inputs, outputs=outputs, name="swap_prediction")
        model.summary()

        return model

    def model_train(self, circuit, model):
        # TODO: split data into test and train data
        # TODO: reshape the data
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam,
                      metrics=["accuracy"]
                      )

        # TODO: model.fit()
        # TODO: model.evualate()
        # TODO: print loss and accuracy results
        return model

    def schedule_gate(self, connectivity, gate):
        """
        Adds gate to self if gate is compatible with connectivity, otherwise returns False, indication to perform MCTS
        """
        if environment.is_in_connectivity(gate, connectivity):
            self.scheduled_gates.append(gate)
        else:
            self.add_swap(gate)


    def add_swap(self, gate):
        """
       Action from MCTS added to the scheduled gates
       """
        #TODO: get this import working
        #mcts = monte_carlo_ts.MCTS(self, )
        #action = mcts.mcts(gate)
        #for i in action:
        #    self.scheduled_gates.append(i)
        print(self.scheduled_gates)

    # TODO: print out tree of MCTS
    def show_tree(self):
        pass


if __name__ == "__main__":

    c = Circuit(4)
    all = Allocation()
    con = all.connectivity()
    circ = c.get_circuit()

    a = Agent()
    for i in circ:
        # print(i)
        a.schedule_gate(con, i)
