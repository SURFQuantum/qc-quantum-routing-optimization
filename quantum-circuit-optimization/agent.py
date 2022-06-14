import pickle
from os.path import exists

from keras.optimizer_v2.adam import Adam
from keras.models import model_from_json
import environment
import numpy as np
from keras.layers import Dense, Reshape, Softmax
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential, save_model, load_model
from keras.optimizers import adam_v2
import os

from circuit import Circuit
from monte_carlo_ts import MCTS
from save_data import load_object, save_circuit
from keras.layers.core import Dense
import qubit_allocation
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


    def circuit_length(self, circuit):
        self.length = len(circuit)

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
        0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

        output = [ 0.  1.  0. 1.]
        """

        # numpy vector full of zeros
        # state = np.zeros(self.n_qubits * ((len(scheduled_gates)) + (self.n_qubits - 1)) + 1)
        state = np.zeros(40)
        q = []

        # Creating matrix with the position numbers, timestep on the columns
        # and qubits on the rows (first matrix from the comments above)
        for i in range(self.n_qubits):
            qu = []
            for j in range(0,len(state), self.n_qubits):
                qu.append(j+i)
            q.append(qu)
        # print(q)

        # check in what row the qubits are
        # print(f'scheduled gate are {scheduled_gates}')
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
        return state


class Agent:

    def __init__(self, circuit, MCTS, all):
        self.learning_rate = 0.01
        self.scheduled_gates = []
        self.n_qubits = circuit.n_qubits
        self.input_size = self.n_qubits * (self.n_qubits - 2) + 1
        self.epochs = 50
        self.mcts = MCTS
        self.connectivity = all.connectivity()
        self.swap = False
        self.logic = [0,1,2,3]


    def build_model(self, input_size):

        model = Sequential()

        model.add(Dense(10, input_dim=input_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(8, activation='linear'))
        model.add(Reshape((1, 4, 2), input_shape=(8,)))
        model.compile(loss=CategoricalCrossentropy(from_logits=True),
                      optimizer=adam_v2.Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self, model):
        filepath = './saved_model'
        save_model(model, filepath)

    def model_train(self, model, state, y_train):
        model.fit(state, y_train, verbose=2, epochs=self.epochs)
        self.save_model(model)

    def schedule_gate(self, gate):
        """
        Adds gate to self if gate is compatible with connectivity, otherwise returns False, indication to perform MCTS
        """
        # while not self.swap:
        if environment.is_in_connectivity(gate, self.connectivity):
            self.scheduled_gates.append(gate)
        else:
            self.swap = True
            schedule = self.scheduled_gates.copy()
            schedule.append(gate)
            # state = self.state.state(schedule)
            swaps, self.connectivity = self.add_swap(gate,self.scheduled_gates)
            #print(swaps)
            for x in swaps:
                self.scheduled_gates.append(x)

        #gate,self.logic = self.mcts.swap_circuit()

        #print(self.scheduled_gates)

    def add_swap(self, gate, state):
        """
       Action from MCTS added to the scheduled gates
       """
        return self.mcts.mcts(gate,state)

if __name__ == "__main__":

    for i in range(50):
        c = Circuit(4)
        all = qubit_allocation.Allocation(c)
        con = all.connectivity()
        topo = all.topology
        s = State(c)
        mcts = MCTS(con,topo, s)
        a = Agent(c,mcts, all)
        circ = c.get_circuit()
        print(f'Begin of the circuit {circ}')
        for i in circ:
            a.schedule_gate(i)
        #print(a.scheduled_gates)

        save_circuit(a.scheduled_gates)
        directory = 'qiskit_depth/'

        for filename in sorted(os.listdir(directory)):
            cir = load_object(directory+filename)
            print(cir)
        print('Circuit fully scheduled')



