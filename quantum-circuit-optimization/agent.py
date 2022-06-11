import keras.losses
import numpy as np
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras.models import model_from_json
from PER_memory_tree import Memory

import environment
# from monte_carlo_ts import MCTS
from environment import Environment
from circuit import Circuit

# class state defines the board and decides reward, end and next position
from qubit_allocation import Allocation

from keras import Input, Model
from keras.layers.core import Dense

import numpy


class State:
    def __init__(self, circuit):
        self.circuit = circuit.get_circuit()  # [[0, 1, 0], [3, 2, 0], [3, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0], [2, 3, 0]]
        self.isEnd = False
        self.length = 0
        self.n_qubits = self.highest()

    def highest(self):
        max = 0
        list = self.circuit

        for i in list:
            for j in i:
                if j > max:
                    max = j

        max += 1
        return max

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

    def check(self, i, q):
        if i in q:
             return True

    def zero_runs(self, a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    def state(self, scheduled_gates):
        # circuit from file [[0, 1, 0], [3, 2, 0], [3, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0], [2, 3, 0]]
        """
          |ts1|ts2|ts3 |ts4| ts5
        ------------------------  #ts
        q0| 0 | 4 | 8 | 12 | 16 | 20
        ------------------------
        q1| 1 | 5 | 9 | 13 | 17
        ------------------------
        q2| 2 | 6 | 10| 14 | 18
        ------------------------
        q3| 3 | 7 | 11| 15 | 19

        schedule_gates = [[0,1,0],[3,2,0],[3,0,0],[0,2,0]]
        CNOT = 1, SWAP = 2

          |ts1 |ts2 |ts3 |ts4| ts5
        ------------------------  #ts
        q0| 11 | 13 | 12 | 0 | 0 | 5
        ------------------------
        q1| 10 | 0  | 0 | 0 | 0
        ------------------------
        q2| 13 | 0  | 10| 0 | 0
        ------------------------
        q3| 12 | 10 | 0| 0  | 0

        state is [11. 10. 13. 12. 13.  0.  0. 10. 12.  0. 10.  0.  0.  0.  0.  0.  0.  0.
        0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  5.]

        output = [ 0.  1.  0. 1.]
        """

        # numpy vector full of zeros
        state = np.zeros(self.n_qubits * ((len(scheduled_gates)) + (self.n_qubits - 1)) + 1)
        q = []

        # Creating matrix with the position numbers, timestep on the columns
        # and qubits on the rows (first matrix from the comments above)
        for i in range(self.n_qubits):
            qu = []
            for j in range(0,len(state), self.n_qubits):
                qu.append(j+i)
            q.append(qu)
        print(q)

        # check in what row the qubits are
        print(f'scheduled gate are {scheduled_gates}')
        for i in scheduled_gates:
            for h in range(self.n_qubits):
                qubit_1 = self.check(i[0], q[h])
                if qubit_1:
                    qubit_1 = q[h]
                    break
            for d in range(self.n_qubits):
                qubit_2 = self.check(i[1], q[d])
                if qubit_2:
                    qubit_2 = q[d]
                    break

            # place the qubit on the right place on the vector, if a vector is already taken, look for the next spot in the row
            for j, k in zip(qubit_1, qubit_2):

                if state[j] == 0 and state[k] == 0:
                    # add 10 to indicate its a CNOT, add 20 to indicate SWAP
                    if i[2] == 0:
                        state[j] = i[1] + 10
                        state[k] = i[0] + 10
                        break
                    if i[2] == 1:
                        state[j] = i[1] + 20
                        state[k] = i[0] + 20
                        break

        # find where in the circuit there are no more gates
        runs = self.zero_runs(state)
        last_item = runs[-1]
        print(f' gates end at {last_item}')

        # calculate number of timesteps
        for i in q:
            ts = self.check(last_item[0],i)
            if ts:
                timestep = i.index(last_item[0]) + 1
                break

        print(f'number of timesteps are {timestep}')
        position = len(state) - 1
        state[position] = timestep
        print(f'the state is {state}')
        return state


class Agent:

    def __init__(self, circuit, state):
        self.state = state
        self.learning_rate = 0.1
        self.scheduled_gates = []
        self.model = None
        self.n_qubits = circuit.n_qubits
        self.input_size = self.n_qubits * (self.n_qubits - 2) + 1
        self.memory_size = 500
        self.memory_tree = Memory(self.memory_size)
        self.target_model = self.build_model()

        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9

    def build_model(self):

        # Input is number of qubits * maximum number swap gates that can be scheduled to decide the dimension of the circuit + the timestep entry
        #
        # input_size = 32
        #
        # model = Sequential()
        # model.add(Embedding(input_size, 32, input_length=10))
        # model.add(Dense(32, input_dim=input_size, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(8, activation='linear'))
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))

        pass
    def save_model(self, model_name=None):
        # Serialize model to JSON
        model_json = self.current_model.to_json()

        if model_name is not None:
            filepath = "./models/" + model_name
        else:
            filepath = "./models/agent_model"

        with open(filepath + ".json", "w") as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        self.current_model.save_weights(filepath + ".h5")
        print("Saved model to disk")

    def load_model(self, model_name=None):
        self.epsilon = self.epsilon_min

        if model_name is not None:
            filepath = "./models/" + model_name
        else:
            filepath = "./models/agent_model"

        # Load json and create model
        json_file = open(filepath + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.current_model = model_from_json(loaded_model_json)

        # Load weights into new model
        self.current_model.load_weights(filepath + ".h5")
        self.current_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print("Loaded model from disk")

    def remember(self, state, reward, next_state, done):
        """
        Store experience in the memory tree
        """
        self.memory_tree.store((state, reward, next_state, done))

    def update_target_model(self):
        """
        Copy weights from the current model to the target model
        """
        self.target_model.set_weights(self.current_model.get_weights())

    def get_quality(self, current_state, next_state, action_chooser='model'):
        neural_net_input = self.get_NN_input(current_state, next_state)

        if action_chooser == 'model':
            Qval = self.current_model.predict(neural_net_input)[0]
            Qval = Qval.reshape(4,2)
        elif action_chooser == 'target':
            Qval = self.target_model.predict(neural_net_input)[0]
            Qval = Qval.reshape(4, 2)

        return Qval
    def model_train(self):

        pass

    def schedule_gate(self, connectivity, gate):
        """
        Adds gate to self if gate is compatible with connectivity, otherwise returns False, indication to perform MCTS
        """
        if environment.is_in_connectivity(gate, connectivity):
            self.scheduled_gates.append(gate)
        else:
            schedule = self.scheduled_gates
            schedule.append(gate)
            self.state.state(schedule)

    def add_swap(self, gate):
        """
       Action from MCTS added to the scheduled gates
       """

        # TODO: get this import working
        # mcts = monte_carlo_ts.MCTS(self, )
        # action = mcts.mcts(gate)
        # for i in action:
        #    self.scheduled_gates.append(i)
        print(self.scheduled_gates)

    # TODO: print out tree of MCTS
    def show_tree(self):
        pass


# if __name__ == "__main__":
#
#     c = Circuit(4)
#     all = Allocation(c)
#     con = all.connectivity()
#
#     circ = c.get_circuit()
#     s = State(c)
#     a = Agent(c,s)
#     a.build_model()
