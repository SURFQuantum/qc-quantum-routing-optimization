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

    def get_out_degrees(self, graph):
        """
        Calculates out-degrees for each node in the graph represented as a dictionary.
        Returns a dictionary mapping nodes to their out-degrees, sorted in descending order.
        """
        out_degrees = {}
        for node, connections in graph.items():
            out_degrees[node] = len(connections)
        out_degrees = {k: v for k, v in sorted(out_degrees.items(), key=lambda item: item[1], reverse=True)}
        return out_degrees

    def find_qubit_mapping(self, connectivity_set):
        """
        Find a valid qubit mapping given a connectivity set and hardware topology.

        Args:
            connectivity_set (list): List of tuples representing the desired qubit connectivity.

        Returns:
            dict: A valid qubit mapping as a dictionary with connectivity set indices as keys and
                  hardware topology indices as values, or None if no valid mapping exists.
        """

        # Create an adjacency list from connectivity set
        connectivity_graph = {}

        # hardware_topology = self.connectivity(hardware='linear')
        # hardware_topology (dict): Hardware topology represented as an adjacency list.
        hardware_topology = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        for edge in connectivity_set:
            if edge[0] not in connectivity_graph:
                connectivity_graph[edge[0]] = set()
            if edge[1] not in connectivity_graph:
                connectivity_graph[edge[1]] = set()
            connectivity_graph[edge[0]].add(edge[1])
            connectivity_graph[edge[1]].add(edge[0])

            print(f'connectivity graph {connectivity_graph}')

        # Create an adjacency list from hardware topology
        hardware_graph = {}
        for node in hardware_topology:
            hardware_graph[node] = set(hardware_topology[node])

            print(f'hardware graph {hardware_graph}')

        # Initialize qubit mapping and visited set
        qubit_mapping = {}
        visited = set()

        outdegree_con = self.get_out_degrees(connectivity_graph)
        outdegree_hard = self.get_out_degrees(hardware_graph)

        print(f'outdegree_con {outdegree_con}')
        print(f'outdegree_hard {outdegree_hard}')

        for qubit, connections in connectivity_set:
            matching_qubit = None
            outdegree_conn_qubit = outdegree_con[qubit]
            for matching_qubit2 in hardware_topology:
                if matching_qubit2 not in qubit_mapping.values() and (
                        matching_qubit is None or outdegree_hard[matching_qubit2] >= outdegree_hard[
                    matching_qubit]) and outdegree_hard[matching_qubit2] >= outdegree_conn_qubit:
                    if matching_qubit2 in connectivity_set and (
                            matching_qubit is None or matching_qubit2 == qubit_mapping[qubit]):
                        matching_qubit = matching_qubit2
                    elif matching_qubit2 not in connectivity_set:
                        matching_qubit = matching_qubit2
            if matching_qubit is not None:
                qubit_mapping[qubit] = matching_qubit

        # Find valid mapping for each edge in connectivity set
        # for edge in connectivity_set:
        #     start, target = edge
        #     if start not in visited:
        #
        #         outdegree = self.get_out_degrees(hardware_graph)
        #         if outdegree == -1:
        #             return None
        #         # Find the qubit in the hardware topology with the shortest distance to the target
        #         qubit_candidates = []
        #         for qubit in hardware_topology:
        #             if self.get_out_degrees(connectivity_graph) == outdegree:
        #                 qubit_candidates.append(qubit)
        #         if not qubit_candidates:
        #             return None
        #         # Choose the qubit with the lowest index as the mapping
        #         qubit_mapping[start] = min(qubit_candidates)
        #         visited.add(start)

        return qubit_mapping

    def connectivity(self, hardware):

        # TODO: rewrite this in something like hardware_topology = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        # Connectivity of the physical qubit in hardware
        # keys are indices of the physical qubits
        # values is a list of neighboring physical qubits that each q is connected to

        # Example: {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        # Physical q0 connected to physical q1
        # Physical q1 connected to physical q0 & q2
        # Physical q2 connected to physical q1 & q3
        # Physical q3 connected to physical q2

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
                    # [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        elif hardware == "ibm-tokyo":
            connectivity_set = [(1, 7), (2, 6), (3, 9), (4, 8), (5, 11), (6, 10), (7, 13), (8, 12), (11, 17), (12, 16),
                                (13, 19), (14, 18)]

        elif hardware == "sycamore":
            # TODO: sycamore adjacency matrix
            return 0

        elif hardware == "grid":
            # TODO: grid adjacency matrix
            return 0
        return


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
