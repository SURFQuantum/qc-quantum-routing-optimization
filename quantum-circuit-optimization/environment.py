import qubit_allocation
from qubit_allocation import Allocation
from circuit import Circuit


class Environment:

    def __init__(self, allocation_class, circuit_class):
        self.qubits = allocation_class.qubits
        # print(self.qubits)
        self.topology = allocation_class.topology
        # print(self.topology)
        self.circuit = circuit_class.get_circuit()  # [(0, 1), (3, 2), (3, 0), (0, 2), (1, 2), (1, 0), (2, 3)]
        # print(self.circuit)
        self.connectivity = allocation_class.connectivity()  # [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
        # print(self.connectivity)

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

        # [0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0]

        circuit_matrix = [0] * (pow(self.qubits, 2))
        count = 0
        connectivity = sorted(self.connectivity, key=lambda x: x[0], reverse=False)

        #TODO: fix circuit matrix
        # for i, obj in enumerate(connectivity):
        #     for j in range(len(connectivity)):
        #         if obj[0] == i:
        #             if obj[1] == i+1:
        #                 print(obj)
        #                 break
        #         else:
        #             print(i)
        return


b = Allocation()
b.qubit_allocation

c = Circuit(4)
a = Environment(b, c)

a.circuit_matrix()
