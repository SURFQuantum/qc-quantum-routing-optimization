import random
from qiskit import QuantumCircuit


class Circuit:

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []

    def generate_completely_random_circuit(self, n_qubits, n_gates):

        self.n_qubits = n_qubits
        circuit = QuantumCircuit(n_qubits)

        for _ in range(n_gates):
            q1 = random.randint(0, n_qubits - 1)
            q2 = random.randint(0, n_qubits - 1)

            while q1 == q2:
                q1 = random.randint(0, n_qubits - 1)
                q2 = random.randint(0, n_qubits - 1)

            circuit.cnot(q1, q2)

        return circuit

    @staticmethod
    def from_gates(n_qubits, gates):
        circuit = QuantumCircuit(n_qubits)
        circuit.gates.extend(gates)
        return circuit

    ### GATES ###

    def cnot(self, q1, q2):
        if q1 >= self.n_qubits or q2 >= self.n_qubits:
            raise Exception('Tried to add a gate ' + str((q1, q2)) + \
                            ' but circuit only has ' + str(self.n_qubits) + ' qubits')

        self.gates.append((q1, q2))

    ### OTHER METHODS ###

    def depth(self):
        d = [0] * self.n_qubits

        for (q1, q2) in self.gates:
            d_max = max(d[q1], d[q2])

            d[q1] = d_max + 1
            d[q2] = d_max + 1

        return max(d)

