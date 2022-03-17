import random
from qiskit import QuantumCircuit
import os
import warnings



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
        circuit = Circuit(n_qubits)
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

    # TODO: remove the random qubit allocation and fix that
    def get_circuit(self):
        ## circuit:

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        directory_path = './'

        files = os.listdir(directory_path)
        qasm_files = list(filter(lambda file_name: len(file_name) > 5 and file_name[-5:] == ".qasm", files))

        circuits = []

        for i, file_name in enumerate(qasm_files):
            file_path = directory_path + file_name

            if os.path.getsize(file_path) > 10000:
                continue

            qiskit_circuit = QuantumCircuit.from_qasm_file(file_path)

            gates = []

            for gate_obj, qubits, _ in qiskit_circuit.data:
                if len(qubits) > 1:
                    if gate_obj.__class__.__name__ not in ["CnotGate", "CXGate"]:
                        exit("Non-cnot gate (" + gate_obj.__class__.__name__ + ") found for circuit: " + str(file_name))

                    gate = (qubits[0].index, qubits[1].index)

                    gates.append(gate)
                    #[(0, 1), (3, 2), (3, 0), (0, 2), (1, 2), (1, 0), (2, 3)]

            circuit = Circuit.from_gates(16, gates)
            circuits.append(circuit)

        return list(filter(lambda c: c.depth() < 200, circuits))

    def circuit_matrix(self, circuit):
        return


circ = Circuit(16)
circuits = list(filter(lambda c: c.depth() < 100, circ.get_circuit()))

for circuit in circuits:
    print('Circuit depth:', circuit.depth())
    max_qubits = [max(q1, q2) for (q1, q2) in circuit.gates]
    print('Max qubit used:', max(max_qubits)+1)
    print()
