import keras.losses
import numpy as np
from keras.optimizer_v2.adam import Adam

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
    def __init__(self, circuit, agent):
        self.circuit = circuit.get_circuit()  # [[0, 1, 0], [3, 2, 0], [3, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0], [2, 3, 0]]
        self.isEnd = False
        self.length = 0
        self.scheduled_gates = agent.scheduled_gates
        self.n_qubits = self.highest()


    def highest(self):
        max = 0
        list = self.circuit

        for i in list:
            for j in i:
                if j > max:
                    max = j

        max +=1
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

    def check(self, i, q0, q1, q2, q3):
        if i in q0:
            return q0
        elif i in q1:
            return q1
        elif i in q2:
            return q2
        else:
            return q3

    def state(self):
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
        q0| 10 | 13 | 12 | 0 | 0 | 5
        ------------------------
        q1| 11 | 0  | 0 | 0 | 0
        ------------------------
        q2| 13 | 0  | 10| 0 | 0
        ------------------------
        q3| 12 | 10 | 0| 0  | 0
        """

        # numpy vector full of zeros
        state = np.zeros(self.n_qubits * ((len(self.scheduled_gates)) + (self.n_qubits-1)))

        # index number that are connected to the qubits
        q0 = [0,4,8,12,16,20,24,28]
        q1 = [1,5,9,13,17,21,25,29]
        q2 = [2,6,10,14,18,22,26,30]
        q3 = [3,7,11,15,19,23,27,31]

        # check in what row the qubits are
        print(self.scheduled_gates)
        for i in self.scheduled_gates:
            print(f'gate {i}')
            qubit_1 = self.check(i[0], q0,q1,q2,q3)
            qubit_2 = self.check(i[1], q0, q1, q2, q3)
            #place the qubit on the right place on the vector, if a vector is already taken, look for the next spot in the row
            for j in qubit_1:
                if state[j] == 0:
                    # add 10 to indicate its a CNOT, add 20 to indicate SWAP
                    if i[2] == 0:
                        print(j)
                        state[j] = i[1]+10
                        print(f' {i[1]} qubit komt op state {j}')
                        break
                    if i[2] ==1:
                        state[j] = i[1] + 20
                        break
            for j in qubit_2:
                if state[j] == 0:
                    if i[2] == 0:
                        state[j] = i[0] + 10
                        print(f' {i[0]} qubit komt op state {j}')
                        break
                    if i[2] == 1:
                        state[j] = i[0] + 20
                        break
        # TODO: start is right, but the rest isnt completely correct, i need to say q1 location > q0 location -1
        # should be [11,10,13,12,0,12,11,10,13,12,0,0..]
        print(state)
        return state

class Agent:

    def __init__(self, circuit):
        self.learning_rate = 0.1
        self.scheduled_gates = []
        self.model = None
        self.n_qubits = circuit.n_qubits
        self.input_size = self.n_qubits*(self.n_qubits-2)+1


    def build_model(self):

        # Input is number of qubits * maximum number swap gates that can be scheduled to decide the dimension of the circuit + the timestep entry

        inputs = Input(shape=(self.input_size,))
        dense = Dense(8, activation="relu")
        x = dense(inputs)
        x = Dense(8, activation="relu")(x)
        outputs = Dense(10)(x)

        model = Model(inputs=inputs, outputs=outputs, name="swap_prediction")
        model.summary()

        return model

    def model_train(self):
        model = self.build_model()
        state = State.state()
        #TODO: reshape circuit into state representation

        x_train = state[:80]
        y_train = state[:80]

        x_test = state[80:]
        y_test = state[80:]

        x_train = x_train.reshape(600, self.input_size).astype("float32") / 255
        x_test = x_test.reshape(100, self.input_size).astype("float32") / 255

        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=Adam,
                      metrics=["accuracy"]
                      )
        model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
        test_scores = model.evaluate(x_test, y_test, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

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

    a = Agent(c)
    for i in circ:
        # print(i)
        a.schedule_gate(con, i)
    s = State(c,a)
    s.state()
