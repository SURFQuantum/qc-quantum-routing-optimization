import os
import numpy as np
import random
import cirq
from cirq import Circuit, NamedQubit, CNOT, LineQubit, Moment
from typing import List, Tuple

from state import Circuit, Topology
import networkx as nx

class Environment:

    def __init__(self, 
                 circuit: Circuit, 
                 target_topology: Topology,
                 distance_metric: str):
    
        self.circuit = circuit
        self.topology = target_topology
        self.distance_metric = distance_metric

    def perform_swap(self, swap_qubits: Tuple[int, int]):

        # We are converting from a networkx graph to numpy and then back to nx
        # TODO: find a way to make this more efficient?
        topology = self.circuit.topology
        topology[swap_qubits] = topology[swap_qubits[1], swap_qubits[0]]


    def get_reward(self):
        raise NotImplementedError
    
    def get_DQN_input(self):
        raise NotImplementedError

    def is_swap_possible(self, swap_qubits: Tuple[int, int]) -> bool:
        return self.topology.adjacency_topology[swap_qubits]
