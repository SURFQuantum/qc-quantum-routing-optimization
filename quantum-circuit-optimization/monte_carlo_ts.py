import itertools
from math import log, sqrt, e, inf
from sre_parse import State

import random
from agent import Agent
from circuit import Circuit
from qubit_allocation import Allocation


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class Node:
    def __init__(self):
        self.reward = 0
        self.parent_visits = 1
        self.child_visits = 1
        self.children: list[Node] = []
        self.parent = None
        self.expanded = False
        self.action = None
        self.ucb = 0
        self.cnot = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def update_win_value(self, value):
        self.reward += value
        self.child_visits += 1

        if self.parent:
            self.parent.update_win_value(value)

class MCTS:

    def __init__(self, circuit, allocation):
        # print(f'the state is {self.state}')# a circuit [[0,1,0],[1,0,0]] from first action
        # self.constraints = allocation_class.connectivity()
        self.reward = 0
        # self.action = 0  #
        self.parent = 0
        self.N = 0
        self.n = 0
        self.n_qubits = circuit.n_qubits
        #self.n_qubits = 6
        self.root = Node()
        self.connectivity = allocation.connectivity()
        self.node_evaluator = lambda child, montecarlo: None

    # Node is a circuit with gates, parent node should change for every action, child node is the possible action
    # coming from parent node
    @property
    def action(self):
        possible_action = []
        for i, j in pairwise(list(range(self.n_qubits))):
            swap_gate = [i, j, 1]
            possible_action.append(swap_gate)

        # print(possible_action)
        return possible_action

    def ucb(self, node_i):
        """
        Upper Confidence Bound for selecting the best child node
        """
        ucb = node_i.reward + sqrt(2) * (sqrt(log(node_i.parent_visits + e + (10 ** (-6))) / (node_i.child_visits + 10 ** (-1))))
        return ucb

    def swap_schedule(self, i, end_state, gate):

        a, b, _ = i
        end_distance = inf
        #print(f'gate is {gate}')

        # CNOT-gate
        new_gate = [gate[0], gate[1]]

        # Swap the nodes and change the CNOT-gate
        for x in range(len(new_gate)):
            if new_gate[x] == a:
                new_gate[x] = b
            elif new_gate[x] == b:
                new_gate[x] = a
        new_gate.append(0)

        #print(f'new gate is {new_gate}')
        #print(f' new CNOT-gate position {new_gate}')

        # calculate the distance to an operable qubit connectivity location
        for i in self.connectivity:
            q0 = new_gate[0] - i[0]
            q1 = new_gate[1] - i[1]

            if q0 < 0:
                q0 = q0*-1

            if q1 < 0:
                q1 = q1*-1
            distance = q0 + q1
            if distance < end_distance:
                end_distance = distance

        #Reward for improving the CNOT
        if end_distance == 0:
            reward = 100
            end_state = True
        elif end_distance < 4:
            reward = 5
        else:
            # if distance to an operable qubit location is more than 4, then this swap location is not recommended
            reward = -1

        #print(reward)
        return end_state, reward, new_gate

    def selection(self, gate):
        # receives iteration
        # choosing child node based on Upper Confidence Bound
        """
        Iterate through all the child of the given state and select the one with highest UCB value
        """
        circuit = []
        action = self.action
        self.root.action = gate
        end_state = False
        timestep = 0
        N = 3 #number of max iterations

        # Iterate through the children until the CNOT is operable
        while not end_state:
            timestep += 1
            for i in action:
                child = Node()
                child.action = i
                end_state, reward, new_gate = self.swap_schedule(child.action, end_state, gate)
                child.ucb = self.ucb(child)
                child.cnot = new_gate
                if end_state:
                    child.reward += reward
                    child.child_visits = timestep
                    self.root.add_child(child)
                    break
                child.reward = reward
                child.child_visits = timestep

                self.root.add_child(child)

            # Find the best child
            child = self.select_child(self.root)
            gate = child.cnot
            self.root = child

            circuit.append(child.action)
            # Not more than 6 iterations for selection
            if timestep == N:
                break

        circuit.append(gate)
        print(f' Circuit is {circuit}')
        return circuit, child

    # function for the result of the simulation
    def expand(self, root):
        print(root.reward)
        if root.reward != 100:
            end_state = False
            random_node = Node()
            random_node.action = root.action
            while random_node.action == root.action:
                random_node.action = random.choice(self.action)

            random_node.ucb = self.ucb(random_node)
            _, reward, new_gate = self.swap_schedule(random_node.action, end_state, root.cnot)

            random_node.reward = reward
            random_node.cnot = new_gate

            self.simulate(random_node)
            return random_node
        else:
            return None

    # function for backpropagation
    def backpropagation(self):
        # if is_root(node) return
        # node.stats = update_stats(node, result)
        # backpropagation(node.parent)
        pass

    def select_child(self, root):
        best_children = []
        best_score = float('-inf')

        for child in root.children:
            score = child.ucb
            #print(score)

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        select = root.action
        while select == root.action:
            child_node = random.choice(best_children)
            select = child_node.action
        return child_node

    def simulate(self, root):
        pass

    def mcts(self, gate):
        circuit, child = self.selection(gate)
        expansion_node = self.expand(child)
        if expansion_node is not None:
            circuit[-1] = expansion_node.action
            circuit.append(expansion_node.cnot)
        print(circuit)
        self.backpropagation()
        return circuit

c = Circuit(4)
s = State()
print(s)
all = Allocation(c)
con = all.connectivity()
circ = c.get_circuit()

a = Agent(c,s)
gate = (3,0,0)

m = MCTS(c,all)
m.mcts(gate)
# while True:
#     broken_gate = a.schedule_gate(con, circ)
#     if broken_gate is None:
#         break
#     m = MCTS(c, all)
#     gate_to_fix = m.mcts(broken_gate)
#     a.add_swap(gate_to_fix)
