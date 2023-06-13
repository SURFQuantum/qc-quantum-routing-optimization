import numpy as np
from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt

def convert_list_to_adjacency(qubit_gates: List[Tuple[int, int]], num_qubits: int) -> np.ndarray:
    
    state_adjacency = np.zeros((num_qubits, num_qubits))
    qubit_gates = np.array(qubit_gates)
    state_adjacency[qubit_gates] = 1

    return state_adjacency

def show_graph(graph: nx.Graph) -> None:
    nx.draw(graph)
    plt.draw()
    plt.show()

