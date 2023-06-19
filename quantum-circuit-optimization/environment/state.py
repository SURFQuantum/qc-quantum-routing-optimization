import os
import numpy as np
import random
import cirq
from cirq import Circuit, NamedQubit, CNOT, SWAP, LineQubit, Moment
from cirq.contrib.routing import get_circuit_connectivity
from utils import convert_list_to_adjacency, show_graph
from typing import List, Tuple, Union
import networkx as nx

class TimeState():
    """
    Represents a state with gates for a certain time step in a circuit.

    Args:
        gates (List[Tuple[int, int]]): List of gate pairs, where each pair represents a CNOT gate acting on qubits.
        time_step (int): current Time step.
        num_qubits (int, optional): Number of qubits in the quantum system. Defaults to 4.

    Attributes:
        adjacency_gates np.ndarray: Adjacency matrix representing the connectivity of gates.
        cirq_timestep (Circuit, optional): cirq operations representing the gates.

    """

    def __init__(self, gates: List[Tuple[int,int]], 
                 time_step: int,
                 num_qubits: int=4,
                 gate_type: str="CNOT"):
        
        self.gates = gates
        self.time_step = time_step
        self.gate_type = gate_type
        
        self.num_qubits = num_qubits
        self.adjacency_gates = convert_list_to_adjacency(self.gates, self.num_qubits)
        self.cirq_timestep = self.cirq_operations()

    def cirq_operations(self) -> List:
        """
        Convert the gates to cirq operations.

        Returns:
            list of cirq operations representing the gates in current time step.

        """

        q = LineQubit.range(self.num_qubits)

        # assume we are only dealing with CNOTs at the start
        if self.gate_type=="CNOT":
            ops = [CNOT(q[gate[0]], q[gate[1]]) for gate in self.gates]

        elif self.gate_type=="SWAP":
            ops = [SWAP(q[gate[0]], q[gate[1]]) for gate in self.gates]
        
        return ops
    
    def __str__(self):
        print("\nCircuit:\n")
        circuit_state = Circuit([Moment(*self.cirq_timestep)])
        return str(circuit_state)

class TopologyState():
    """
    Represents the topology of a quantum circuit. This can be a topology of a current circuit or 
    for the target topology. We include a few distance measures to compare both.

    Args:
        topology (np.ndarray): Adjacency matrix representing the circuit topology.
        num_qubits (int): Number of qubits in the quantum circuit.

    Attributes:
        topology (nx.Graph): NetworkX graph representing the circuit topology.
        num_qubits (int): Number of qubits in the quantum circuit.

    """

    def __init__(self,
                 topology: Union[List, nx.Graph]):
        
        if isinstance(topology, nx.Graph):
            self.nx_topology = topology
            self.adjacency_topology = nx.to_numpy_array(topology)
            self.num_qubits = self.adjacency_topology.shape[0]

        elif isinstance(topology, list):
            self.num_qubits = np.unique([qubit for tup in topology for qubit in tup]).shape[0]
            self.adjacency_topology = convert_list_to_adjacency(topology, self.num_qubits)
            self.nx_topology = nx.from_numpy_array(self.adjacency_topology)

    def update(self, topology: np.ndarray) -> None:
        self.nx_topology = nx.from_numpy_array(topology)
        self.adjacency_topology = nx.to_numpy_array(self.nx_topology)

    def hamming_distance_to_circuit(self, topology: nx.Graph) -> int:
        """
        Compute the Hamming distance between the topology and the connectivity of a given circuit.

        *Note: not sure if this simple metric will work with RL at all.

        Args:
            circuit (CircuitState): CircuitState object representing the quantum circuit.

        Returns:
            int: Hamming distance between the topology and the circuit connectivity.

        """
        string1 = self.adjacency_topology.flatten()
        string2 = nx.to_numpy_array(topology).flatten()

        return self.hamming_distance(string1, string2)

    def hamming_distance(self, string1: np.array, string2: str) -> int:
        assert len(string1)==len(string2)

        bitarray1 = np.array(string1, dtype=np.uint8)
        bitarray2 = np.array(string2, dtype=np.uint8)

        distance = np.count_nonzero(bitarray1 != bitarray2)

        return distance
    
    def floyd_warshall(self, topology: nx.Graph) -> np.ndarray:
        """
        Compute the shortest path distances between all pairs of nodes using the Floyd-Warshall algorithm.

        Args:
            topology (nx.Graph): NetworkX graph representing the circuit topology.

        Returns:
            np.ndarray: Array of shortest path distances between all pairs of nodes.

        """
        return nx.floyd_warshall_numpy(topology)
    
    def floyd_warshall_distance_to_circuit(self, topology: nx.Graph) -> int:
        circuit_distances = self.floyd_warshall(topology)
        topology_distances = self.floyd_warshall(self.nx_topology)

        absolute_distance = np.abs(circuit_distances - topology_distances)
        return np.sum(absolute_distance)
        
    def draw(self):
        
        show_graph(self.nx_topology, self.num_qubits)

class CircuitState():
    """
    Represents a quantum circuit state. This is essentially a collection of TimeState objects.

    Args:
        circuit (List[TimeState]): List of TimeState objects representing the circuit at different time steps.
        topology Topology: Adjacency matrix representing the current circuit topology. This gets updated with
                                each swap.
        num_qubits (int, optional): Number of qubits in the quantum circuit. Defaults to 4.

    Attributes:
        cirq_circuit (Circuit): cirq representation of the quantum circuit.

    """

    def __init__(self, circuit: List, 
                 num_qubits: int=4):
        
        self.circuit = circuit
        self.num_qubits = num_qubits
        self.cirq_circuit = self.update_curcuit(self.circuit,"CNOT")
        self.topology = self.get_circuit_connectivity()
        self.time_step = 0

    def length(self) -> int:
        return len(self.circuit)

    def update_curcuit(self, circuit: List, gate: str) -> Tuple[Circuit, TopologyState]:
        self.circuit = [TimeState(timestep, i, self.num_qubits, gate) 
                for i, timestep in enumerate(circuit)]
        
        cirq_circuit = self.circuit_to_cirq(self.circuit)

        return cirq_circuit
    
    def circuit_to_cirq(self, circuit:List[TimeState]) -> Circuit:
            """
            Convert the quantum circuit to Cirq circuit representation.

            Args:
                circuit (List[TimeState]): List of TimeState objects representing the circuit at different time steps.

            Returns:
                Circuit: cirq representation of the quantum circuit.

            """

            circuit = [Moment(*timestep.cirq_timestep) for timestep in circuit]
            circuit = Circuit(circuit)
            return circuit
    
    def insert_circuit(self, index: int, qubits: Tuple[int, int], gate: str) -> None:

        timestate = TimeState(qubits, index, self.num_qubits, gate)
        self.circuit.insert(index, timestate)
        self.cirq_circuit.insert(index, Moment(*timestate.cirq_timestep))
    
    def add_to_cirq(self, timesteps: List, gate: str) -> None:
        """
        Add additional time steps to the Cirq circuit representation. 
        We use EARLIEST Insert Strategy for now. 

        TODO: check other strategy see what's the difference.

        Args:
            timesteps (List): List of time steps to be added to the circuit.

        Returns:
            None

        """
        timesteps = [TimeState(timestep, i, self.num_qubits, gate) 
                for i, timestep in enumerate(timesteps)]
        
        for timestep in timesteps:
            self.cirq_circuit.append(timestep.cirq_timestep, strategy=cirq.InsertStrategy.EARLIEST)
    
    def update_circuit_topology(self, new_connectivity: TopologyState) -> None:
        """
        This represents the Topology of the current circuit. 

        Args:
            new_connectivity Topology: Topology of the circuit

        Returns:
            None

        """
        self.topology = new_connectivity

    def get_circuit_connectivity(self) -> TopologyState:
        """
        Get the connectivity graph of the quantum circuit.

        Returns:
            Topology: graph representing the connectivity of the quantum circuit.

        """

        circuit_graph = TopologyState(get_circuit_connectivity(self.cirq_circuit))
        return circuit_graph

    def __str__(self):
        print("\nCircuit:\n")
        return str(self.cirq_circuit)

def main():
    circuit = [[(0,1), (2,3)], [(0,2)]]
    
    topology = TopologyState([[0,1],[1,2],[2,3]])
    circuit = CircuitState(circuit)
    print(circuit)
    circuit.get_circuit_connectivity().draw()

    circuit.add_to_cirq([[(0,2)], [(0,1)], [(1,3)], [(3,2)]])
    print(circuit)
    circuit.get_circuit_connectivity().draw()
    circuit.add_to_cirq([[(0,1)]])
    print(circuit)
    circuit.get_circuit_connectivity().draw()
    circuit.add_to_cirq([[(2,3)]])
    print(circuit)
    circuit.get_circuit_connectivity().draw()
    circuit.add_to_cirq([[(0,3)]])
    print(circuit)
    circuit.get_circuit_connectivity().draw()
    circuit.add_to_cirq([[(1,2)]])
    print(circuit)
    circuit.get_circuit_connectivity().draw()
    
    print("hamming distance: ", topology.hamming_distance_to_circuit(circuit.topology.nx_topology))
    print("Floyd warshall distance between topologies: ", topology.floyd_warshall_distance_to_circuit(circuit.topology.nx_topology))

if __name__ == "__main__":
    main()
