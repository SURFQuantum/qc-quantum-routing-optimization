import random
from collections import Counter
from itertools import combinations

from circuit import Circuit


class Allocation:

    def __init__(self, circuit_class):
        # linear topology i.e. [0,1,2,3]
        # connectivity would be (0,1) (1,0) (1,2) (2,1) (2,3) (3,2)
        self.topology = []
        self.qubits = 4
        self.gates = circuit_class.get_circuit()



    """
    psi = [0, 1], [3, 2], [3, 0], [0, 2], [1, 2], [1, 0], [2, 3]]
    Weighted Graph of circuit
        q1 
       /  \
    2 /    \ 1
     /   1  \
    q0 ----> q2
     \      /
    1 \    / 2
       \  /
        q3
        
        
    Linear Topology    
    A - B - C - D
    
    Out-degree
    q0 = 3
    q1 = 2
    q2 = 3
    q3 = 2
    
    out degree
    A = 1
    B = 2
    C = 2
    D = 1    
    
    q0 and q2 connected and have highest out-degree, so they should be next to each other on 
    the node with the highest out-degree on the topology
    """

    def weighted_graph(self):
        outdegree_top  = {}
        outdegree_graph = {}
        for i in range(self.qubits):
            outdegree_graph[i] = 0
            outdegree_top[i] = 0
        con = self.connectivity() #[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        gate = [] #[[0, 1], [3, 2], [3, 0], [0, 2], [1, 2], [1, 0], [2, 3]]
        for i in self.gates:
            g = [i[0], i[1]]
            gate.append(g)
        for i in con:
            for j in i:
                if j in outdegree_top: #{0: 2, 1: 4, 2: 4, 3: 2}
                    outdegree_top[j] += 1
        for i in gate:
            for j in i:
                if j in outdegree_graph: #{0: 4, 1: 3, 2: 4, 3: 3}
                    outdegree_graph[j] += 1

    @property
    def qubit_allocation(self):
        # TODO: remove the random qubit allocation and fix that
        # self.topology = (random.sample(range(self.qubits), 4))
        # print(self.topology)

        # self-assigned qubit allocation linearly for testing
        for i in range(self.qubits):
            self.topology.append(i)




    def connectivity(self):
        self.qubit_allocation
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
        #print(connectivity_set)
        return connectivity_set  # [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]


c = Circuit(4)
a = Allocation(c)
a.weighted_graph()
