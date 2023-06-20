import numpy as np
from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt

def convert_list_to_adjacency(qubit_gates: List[Tuple[int, int]], num_qubits: int) -> np.ndarray:
    
    state_adjacency = np.zeros((num_qubits, num_qubits))
    qubit_gates = np.array(qubit_gates)
    state_adjacency[qubit_gates[:,0], qubit_gates[:,1]] = 1
    state_adjacency[qubit_gates[:,1], qubit_gates[:,0]] = 1

    return state_adjacency

def show_graph(graph: nx.Graph, num_qubits: int) -> None:
    pos = nx.spring_layout(graph, seed=0)  # positions for all nodes
    nx.draw_networkx_nodes(graph, nodelist=graph.nodes, pos=pos, node_color="tab:red")
    labels = nx.draw_networkx_labels(graph, pos=pos)    
    nx.draw_networkx_nodes(graph, nodelist=graph.nodes, pos=pos, node_color="green", label=labels)
    nx.draw_networkx_edges(graph, pos=pos, edge_color="blue")

    plt.draw()
    plt.show()


