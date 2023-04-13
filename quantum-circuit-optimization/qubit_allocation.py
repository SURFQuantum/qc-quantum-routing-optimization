import networkx as nx
from circuit import Circuit


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


class Allocation:

    def __init__(self, circuit_class):
        # linear topology i.e. [A,B,C,D,E,F]
        # connectivity would be (0,1) (1,0) (1,2) (2,1) (2,3) (3,2)
        self.topology = []
        self.qubits = 4
        self.gates = circuit_class.get_circuit()
        self.allocation = []

    def bfs_shortest_path_length(self, graph, start, target):
        """
        Find the shortest path length between two nodes in an undirected graph using Breadth-First Search (BFS).

        Args:
            graph (dict): Graph represented as an adjacency list.
            start: Starting node.
            target: Target node.

        Returns:
            int: Shortest path length between start and target, or -1 if no path exists.
        """

        # Check if start and target nodes are in the graph
        if start not in graph or target not in graph:
            return -1

        # Initialize visited set, queue for BFS, and dictionary to store distances
        visited = set()
        queue = [(start, 0)]
        distances = {start: 0}

        # Perform BFS until queue is empty
        while queue:
            node, distance = queue.pop(0)
            visited.add(node)

            # Check if target node is found
            if node == target:
                return distance

            # Explore neighbors of current node
            for neighbor in graph[node]:
                if neighbor not in visited and neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))

        # No path exists between start and target
        return -1

    def find_qubit_mapping(self, connectivity_set):
        """
        Find a valid qubit mapping given a connectivity set and hardware topology.

        Args:
            connectivity_set (list): List of tuples representing the desired qubit connectivity.
            hardware_topology (dict): Hardware topology represented as an adjacency list.

        Returns:
            dict: A valid qubit mapping as a dictionary with connectivity set indices as keys and
                  hardware topology indices as values, or None if no valid mapping exists.
        """

        # Create an adjacency list from connectivity set
        connectivity_graph = {}

        # hardware_topology = self.connectivity(hardware='linear')
        hardware_topology = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        for edge in connectivity_set:
            if edge[0] not in connectivity_graph:
                connectivity_graph[edge[0]] = set()
            if edge[1] not in connectivity_graph:
                connectivity_graph[edge[1]] = set()
            connectivity_graph[edge[0]].add(edge[1])
            connectivity_graph[edge[1]].add(edge[0])

        # Create an adjacency list from hardware topology
        hardware_graph = {}
        for node in hardware_topology:
            hardware_graph[node] = set(hardware_topology[node])

        # Initialize qubit mapping and visited set
        qubit_mapping = {}
        visited = set()

        # Find valid mapping for each edge in connectivity set
        for edge in connectivity_set:
            start, target = edge
            if start not in visited:
                distance = self.bfs_shortest_path_length(hardware_graph, start, target)
                if distance == -1:
                    return None
                # Find the qubit in the hardware topology with the shortest distance to the target
                qubit_candidates = []
                for qubit in hardware_topology:
                    if self.bfs_shortest_path_length(connectivity_graph, start, qubit) == distance:
                        qubit_candidates.append(qubit)
                if not qubit_candidates:
                    return None
                # Choose the qubit with the lowest index as the mapping
                qubit_mapping[start] = min(qubit_candidates)
                visited.add(start)

        return qubit_mapping

    def connectivity(self, hardware):

        # TODO: rewrite this in something like hardware_topology = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        # if not self.topology:
        #     self.qubit_allocation()

        if hardware == "linear":
            connectivity_set = []
            for i, obj in enumerate(self.topology):

                if i == 0:
                    connectivity_set.append((obj, self.topology[i + 1]))
                elif i == (len(self.topology) - 1):
                    connectivity_set.append((obj, self.topology[i - 1]))
                else:
                    connectivity_set.append((obj, self.topology[i - 1]))
                    connectivity_set.append((obj, self.topology[i + 1]))
        elif hardware == "ibm-tokyo":
            connectivity_set = [(1, 7), (2, 6), (3, 9), (4, 8), (5, 11), (6, 10), (7, 13), (8, 12), (11, 17), (12, 16),
                                (13, 19), (14, 18)]

        elif hardware == "sycamore":
            # TODO: sycamore adjacency matrix
            return 0

        elif hardware == "grid":
            # TODO: grid adjacency matrix
            return 0
        return connectivity_set  # [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]

c = Circuit(4)
a = Allocation(c)

##### TEST #####

# Define the logical qubits and physical qubits
logical_qubits = [(0, 1), (1, 0), (1, 2), (2, 1)]

# Allocate logical qubits to physical qubits based on the hardware topology
mapping = a.find_qubit_mapping(logical_qubits)

# Print the resulting mapping
print("Logical qubit to physical qubit mapping:")
print(mapping)
