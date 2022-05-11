import random
from collections import Counter
from itertools import combinations

from circuit import Circuit


class Allocation:

    def __init__(self, circuit_class):
        # linear topology i.e. [A,B,C,D,E,F]
        # connectivity would be (0,1) (1,0) (1,2) (2,1) (2,3) (3,2)
        self.topology = [0,1,2,3,4,5]
        self.qubits = 4
        self.gates = circuit_class.get_circuit()
        self.allocation = []



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
    A - B - C - D - E - F
    
    Out-degree
    q0 = 3
    q1 = 2
    q2 = 3
    q3 = 2
    
    out degree
    A = 1
    B = 2
    C = 2
    D = 2
    E = 2
    F = 1    
    
    q0 and q2 connected and have highest out-degree, so they should be next to each other on 
    the node with the highest out-degree on the topology
    """

    def weighted_graph(self):
        outdegree_top_unsorted  = {}
        outdegree_graph_unsorted = {}
        for i in range(self.qubits):
            outdegree_graph_unsorted[i] = 0
        for i in range(len(self.topology)):
            outdegree_top_unsorted[i] = 0
        con = []
        connect = self.connectivity() #[[0, 1], [1, 2], [2, 3]] unique interactions
        gate = [] #[[0, 1], [3, 2], [3, 0], [0, 2], [1, 2]] unique interactions
        for i in self.gates:
            if [i[1],i[0]] not in gate:
                g = [i[0], i[1]]
                gate.append(g)

        for i in connect:
            if [i[1], i[0]] not in con:
                c = [i[0], i[1]]
                con.append(c)

        for i in gate:
            outdegree_graph_unsorted[i[0]] += 1
            outdegree_graph_unsorted[i[1]] += 1

        for i in con:
            outdegree_top_unsorted[i[0]] += 1
            outdegree_top_unsorted[i[1]] += 1

        # {0: 3, 2: 3, 1: 2, 3: 2}
        outdegree_graph = dict(sorted(outdegree_graph_unsorted.items(),
                                  key=lambda item: item[1],
                                  reverse=True))
        # {1: 2, 2: 2, 0: 1, 3: 1}
        outdegree_top = dict(sorted(outdegree_top_unsorted.items(),
                                      key=lambda item: item[1],
                                      reverse=True))

        return outdegree_graph, outdegree_top

    """
    Initial qubit mapping
    """
    def qubit_allocation(self):
        """
        Random qubit allocation
        """
        # self.topology = (random.sample(range(self.qubits), 4))
        # print(self.topology)

        """
        Linear sequential qubit allocation
        """
        # self-assigned qubit allocation linearly for testing
        # [0, 1, 2, 3, 4, 5]
        # for i in range(self.qubits):
        #     self.topology.append(i)
        #TODO: get the outdegrees to match for the mapping
        outdegree_graph, outdegree_top = self.weighted_graph()


            #print(outdegree_graph[i])
        # for key, value in outdegree_graph.items():
        #     if outdegree_graph[i] == value:
        #         print(i)

        # print(outdegree_top)
        # print(outdegree_graph)


        #TODO: match the out-degrees



    """
    Set of the connectivity 
    """
    def connectivity(self):
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
a.qubit_allocation()
