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

    def get_circuit(self):
        # Turning circuit into a list with gate connectivities

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        directory_path = 'test_circuits/'

        file = os.listdir(directory_path)

        circuits = []

        file_name = 'test_circuits/test.qasm'
        for file_line in file:
            file_path = directory_path + file_line

            if os.path.getsize(file_path) > 10000:
                continue

            # TODO: read lines from file instead of from_qasm_file
            qiskit_circuit = QuantumCircuit.from_qasm_file(file_name)

            for elem in qiskit_circuit:
                print(elem)
            gates = {}

            used_qbits = []

            timestep = {}
            count = 1

            NUM_GATES = 4

            # TODO: fix a format like {1:{(0,1):'CNOT',(3,2):'CNOT'},2:{(0,1):'SWAP,(2,3):'SWAP},3:{},4:{},5:{}} etc
            for gate_obj, qubits, _ in qiskit_circuit.data:
                print(gate_obj, qubits)
                if len(qubits) <= 1:
                    continue

                from qiskit.circuit.library import CXGate
                from sympy.physics.quantum.gate import CNotGate
                from qiskit.circuit.library import SwapGate

                if not isinstance(gate_obj, (CXGate, CNotGate, SwapGate)):
                    raise ValueError(f"Non-CNOT gate {gate_obj.name} found in circuit: {str(file_line)}")

                q1_index, q2_index = qubits[0].index, qubits[1].index
                gate = (q1_index, q2_index)

                if q1_index in used_qbits or q2_index in used_qbits:  # FIXME: check dit even lol
                    count += 1
                    used_qbits = [q1_index, q2_index]
                    gates[gate] = ('SWAP' if isinstance(gate_obj,
                                                        SwapGate) else 'CNOT')  # Implicitly assuming CNOT if not SWAPgate
                    print(used_qbits)
                else:
                    gates[gate] = ('SWAP' if isinstance(gate_obj,
                                                        SwapGate) else 'CNOT')  # Implicitly assuming CNOT if not SWAPgate
                    used_qbits.append(q1_index)
                    used_qbits.append(q2_index)
                    print(used_qbits)

                if len(used_qbits) == NUM_GATES:
                    used_qbits, gates = [], {}
                    count += 1

                timestep[count] = gates

            print(timestep)
            circuit = Circuit.from_gates(4, gates)
            circuits.append(circuit)
        return gates
        # return list(filter(lambda c: c.depth() < 200, circuits))


# Finding out the circuit depth and max qubit used, does into account parallel routing
circ = Circuit(4)
circ.get_circuit()

# circuits = list(filter(lambda c: c.depth() < 100, circ.get_circuit()))
#
# for circuit in circuits:
#     print('Circuit depth:', circuit.depth())
#     max_qubits = [max(q1, q2) for (q1, q2) in circuit.gates]
#     print('Max qubit used:', max(max_qubits)+1)
#     print()
