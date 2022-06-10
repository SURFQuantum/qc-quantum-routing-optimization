import itertools
from typing import List, Tuple, Union

from qubit_allocation import Allocation
from circuit import Circuit


Gate: Union[Tuple[int, int], Tuple[int, int, int]]
class Environment:

    def __init__(self, allocation_class, circuit_class):
        self.qubits = allocation_class.qubits
        # print(self.qubits)
        self.topology = allocation_class.topology
        # print(self.topology)
        self.gates = circuit_class.get_circuit() # [[0, 1, 0], [3, 2, 0], [3, 0, 0], [0, 2, 0], [1, 2, 0], [1, 0, 0], [2, 3, 0]]
        print(self.gates)
        self.connectivity: List[Tuple[int, int]] = allocation_class.connectivity()  # [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]


    # Forgot why I needed this, still keeping it for now
    def circuit_matrix(self):

        #   |q0 |q1 |q2 |q3
        # ------------------
        # q0| 0 | 1 | 0 | 0
        # ------------------
        # q1| 1 | 0 | 1 | 0
        # ------------------
        # q2| 0 | 1 | 0 | 1
        # ------------------
        # q3| 0 | 0 | 1 | 0

        #   |q0 |q1 |q2 |q3
        # ------------------
        # q0| 0 | 1 | 2 | 3
        # ------------------
        # q1| 4 | 5 | 6 | 7
        # ------------------
        # q2| 8 | 9 | 10| 11
        # ------------------
        # q3| 12| 13| 14| 15

        # Sort the connectivities
        connectivity = sorted(self.connectivity, key=lambda x: x[0], reverse=False)
        dict = {}

        # check whether a connection exists
        for i, j in itertools.product(range(self.qubits), repeat=2):
            tup = (i, j)
            dict[tup] = (1 if tup in connectivity else 0)
        print(dict)  # {(0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 0, (1, 0): 1, (1, 1): 0, (1, 2): 1, (1, 3): 0, (2, 0): 0,
                 # (2, 1): 1, (2, 2): 0, (2, 3): 1, (3, 0): 0, (3, 1): 0, (3, 2): 1, (3, 3): 0}

        circuit_vector = list(dict.values())  # [0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0]
        print(circuit_vector)

        return circuit_vector
    #
    # def circuit_connectivity_compare(self, gate):
    #     gate_connection = (gate[0], gate[1])
    # #
    #     return gate_connection in self.connectivity
    #     # if gate_connection in self.connectivity:
    #     #     self.schedule = True
    #     # print(self.schedule)
    #     # return self.schedule



def is_in_connectivity(gate, connectivity):
    gate = (gate[0], gate[1])
    return gate in connectivity
#
# b = Allocation()
# b.qubit_allocation
#
# c = Circuit(4)
# a = Environment(b, c)
# test = [0, 1, 0]
# a.circuit_matrix()
#
# con = b.connectivity()
#
# is_in_connectivity(test, con)


# a.circuit_connectivity_compare(test)
