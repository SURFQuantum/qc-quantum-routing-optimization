import networkx as nx
from circuit import Circuit

class Allocation:

    def __init__(self, circuit_class):
        # linear topology i.e. [A,B,C,D,E,F]
        # connectivity would be (0,1) (1,0) (1,2) (2,1) (2,3) (3,2)
        self.topology = []
        self.qubits = 4
        self.gates = circuit_class.get_circuit()
        self.allocation = []

    def find_best_mapping(self, logical_qubit, physical_qubits, hardware_topology, current_mapping):
        """
        Find the best mapping of a logical qubit to a physical qubit based on the hardware topology,
        considering the current mapping of other logical qubits.

        Args:
            logical_qubit (int): Logical qubit to be mapped.
            physical_qubits (list): List of physical qubits in the hardware topology.
            hardware_topology (nx.Graph): Graph representing the hardware topology.
            current_mapping (dict): Current mapping of logical qubits to physical qubits.

        Returns:
            dict: Mapping of logical qubit to physical qubit.
        """

        best_mapping = None
        best_score = float('inf')

        # Get physical qubits that are already assigned to logical qubits
        assigned_physical_qubits = set(current_mapping.values())

        # Iterate over physical qubits and find the mapping with the minimum distance
        for physical_qubit in physical_qubits:
            if physical_qubit not in assigned_physical_qubits:
                current_mapping[logical_qubit] = physical_qubit
                current_score = 0

                # Compute the distance-based heuristic score for the current mapping
                for neighbor in hardware_topology.neighbors(physical_qubit):
                    if neighbor in physical_qubits:
                        distance = nx.shortest_path_length(hardware_topology, physical_qubit, neighbor)
                        current_score += distance

                # Update the best mapping if the current score is lower
                if current_score < best_score:
                    best_mapping = dict(current_mapping)
                    best_score = current_score

                # Remove the temporary mapping to try the next physical qubit
                del current_mapping[logical_qubit]

        return best_mapping

    def allocate_qubits(self, logical_qubits, physical_qubits, hardware_topology):
        """
        Allocate logical qubits to physical qubits based on the hardware topology.

        Args:
            logical_qubits (list): List of logical qubits to be mapped.
            physical_qubits (list): List of physical qubits in the hardware topology.
            hardware_topology (nx.Graph): Graph representing the hardware topology.

        Returns:
            dict: Mapping of logical qubits to physical qubits.
        """

        # Create an empty mapping of logical qubits to physical qubits
        mapping = {}

        # Iterate over logical qubits and find the best mapping for each
        for logical_qubit in logical_qubits:
            mapping = self.find_best_mapping(logical_qubit, physical_qubits, hardware_topology, mapping)

        return mapping

    """
    Set of the connectivity 
    """

    def connectivity(self):
        if not self.topology:
            self.qubit_allocation()
        topology = self.topology
        connectivity_set = []

        for i, obj in enumerate(topology):

            if i == 0:
                connectivity_set.append((obj, topology[i + 1]))
            elif i == (len(topology) - 1):
                connectivity_set.append((obj, topology[i - 1]))
            else:
                connectivity_set.append((obj, topology[i - 1]))
                connectivity_set.append((obj, topology[i + 1]))
        # print(connectivity_set)
        return connectivity_set  # [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]


def swaps_moving_connectivity(topology):
    connectivity_set = []

    for i, obj in enumerate(topology):
        if i == 0:
            connectivity_set.append((obj, topology[i + 1]))
        elif i == (len(topology) - 1):
            connectivity_set.append((obj, topology[i - 1]))
        else:
            connectivity_set.append((obj, topology[i - 1]))
            connectivity_set.append((obj, topology[i + 1]))
    return connectivity_set


c = Circuit(4)
a = Allocation(c)

##### TEST #####
# Create a hardware topology graph
hardware_topology = nx.Graph()
hardware_topology.add_edges_from([(0, 1), (1, 2), (2, 3)])

# Define the logical qubits and physical qubits
logical_qubits = [0, 1, 2]
physical_qubits = [0, 1, 2, 3]

# Allocate logical qubits to physical qubits based on the hardware topology
mapping = a.allocate_qubits(logical_qubits, physical_qubits, hardware_topology)

# Print the resulting mapping
print("Logical qubit to physical qubit mapping:")
print(mapping)
