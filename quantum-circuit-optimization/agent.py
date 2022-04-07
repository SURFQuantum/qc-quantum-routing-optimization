from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam

import environment
from environment import Environment
from circuit import Circuit

# class state defines the board and decides reward, end and next position
from qubit_allocation import Allocation


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

    def __init__(self):
        self.learning_rate = 0.1
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

    def action(self):
        # while resources_left(time, computational power):
        #     leaf = traverse(root)
        #     simulation_result = rollout(leaf)
        #     backpropagation(leaf, simulation_result)
        #
        # return best_child(root)
        return

    def schedule_gate(self, connectivity, gate):
        """
        Adds gate to self if gate is compatible with connectivity, otherwise returns False, indication to perform MCTS
        """
        if environment.is_in_connectivity(gate, connectivity):
            self.scheduled_gates.append(gate)
        else:
            False

    def add_swap(self, action):
        """
       Action from MCTS added to the scheduled gates
       """
        self.scheduled_gates.append(action)

    # TODO: print out tree of MCTS
    def show_tree(self):
        pass


# if __name__ == "__main__":
    # # create agent for 10,000 episdoes implementing a Q-learning algorithm plot and show values.
    # c = Circuit(4)
    # all = Allocation()
    # con = all.connectivity()
    # circ = c.get_circuit()
    #
    # a = Agent()
    # for i in circ:
    #     # print(i)
    #     a.schedule_gate(con, i)
