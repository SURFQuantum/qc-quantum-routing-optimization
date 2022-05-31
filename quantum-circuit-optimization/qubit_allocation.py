import random
from collections import Counter
from itertools import combinations

import networkx as nx
from networkx.algorithms import isomorphism
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
        for i in range(self.qubits):
            self.topology.append(i)

        #TODO: get the outdegrees to match for the mapping
        outdegree_graph, outdegree_top = self.weighted_graph()
        print(f'weighted graph = {outdegree_graph}')
        print(f'topology = {outdegree_top}')

        # g = {}

        # for key, value in outdegree_graph.items():
        #     for k, v in outdegree_top.items():
        #         if value == v:
        #             if k not in g:
        #                 g[key] = k
        #                 break

        #
        # q0 = q1 = q2 = q3 = q4 = q5= 1
        # a = b = c = d = 1
        # isomorph = 0
        #
        # while isomorph <= 10:
        #     weighted_graph = {'a': [b,c], 'b':[a,d,c], 'c':[b,d,a], 'd':[c,b]}
        #     topology = {'q0':[q1], 'q1':[q0,q2], 'q2':[q1,q3], 'q3':[q2,q4], 'q4':[q3,q5], 'q5':[q4]}
        #     #print(dic1)
        #     #print(dic2)
        #     for i in weighted_graph:
        #
        #         x = tuple(weighted_graph[i])
        #         h = hash(x)
        #         weighted_graph[i] = h
        #
        #     for i in topology:
        #         x = tuple(topology[i])
        #         h = hash(x)
        #         topology[i] = h
        #
        #     a = weighted_graph['a']
        #     b = weighted_graph['b']
        #     c = weighted_graph['c']
        #     d = weighted_graph['d']
        #     #print(f' dict 1{dic1}')
        #
        #     q0 = topology['q0']
        #     q1 = topology['q1']
        #     q2 = topology['q2']
        #     q3 = topology['q3']
        #     q4 = topology['q4']
        #     #print(f' dict 2{dic2}')
        #
        #     count_1 = Counter(weighted_graph.values())
        #     count_2 = Counter(topology.values())
        #     print(f'counts {count_1.values()}')
        #     print(f'counts {count_2.values()}')
        #
        #     print((dict(count_1)).values())
        #
        #     shared_items = {k: count_1[k] for k in count_1 if k in count_2 and count_1[k] == count_2[k]}
        #     # print(len(shared_items))
        #
        #     if count_1 == count_2 or count_2 == count_1:
        #         print('equal')
        #
        #
        #     isomorph+=1
        #
        #
        # G = nx.Graph()
        # H = nx.Graph()
        #
        # G.add_edge(0,4)
        # G.add_edge(0, 1)
        # G.add_edge(0, 2)
        # G.add_edge(2, 3)
        # G.add_edge(3, 4)
        #
        # print(G.number_of_edges())
            # for i in dic2:
            #     for j in i:
            #         print(j)
            #         x = tuple(dic2[j])
            #         h = hash(x)
            #     dic2[i] = h

        #
        # print(dic1)
        # print(dic2)

        # for key, value in outdegree_graph.items():
        #     for k, v in outdegree_top.items():
        #         if k not in g.values():
        #             g[key] = k
        #             break
        #
        #
        # print(f'matching {g}')


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
