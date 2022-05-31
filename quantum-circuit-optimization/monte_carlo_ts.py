import itertools
from math import log, sqrt, e, inf
from sre_parse import State


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
        self.reward = 1
        self.N = 1
        self.n = 0
        self.children: list[Node] = []
        self.parent = None
        self.action = None

class MCTS:

    def __init__(self, agent, circuit, allocation):
        self.state = agent.scheduled_gates
        # print(f'the state is {self.state}')# a circuit [[0,1,0],[1,0,0]] from first action
        # self.constraints = allocation_class.connectivity()
        self.reward = 0
        # self.action = 0  #
        self.parent = 0
        self.N = 0
        self.n = 0
        self.n_qubits = circuit.n_qubits
        self.root = Node()
        self.connectivity = allocation.connectivity()

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
        ucb = node_i.reward + sqrt(2) * (sqrt(log(node_i.N + e + (10 ** (-6))) / (node_i.n + 10 ** (-1))))
        return ucb

    def swap_schedule(self, i, end_state, gate):

        a, b, _ = i
        end_distance = 100
        print(i)

        # CNOT-gate
        new_gate = [gate[0], gate[1]]

        # Swap the nodes and change the CNOT-gate
        for x in range(len(new_gate)):
            if new_gate[x] == a:
                new_gate[x] = b
            elif new_gate[x] == b:
                new_gate[x] = a
        new_gate.append(0)
        print(f' new CNOT-gate position {new_gate}')

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
        return end_state, reward, new_gate, i

    def selection(self, gate):
        # receives iteration
        # choosing child node based on Upper Confidence Bound
        """
        Iterate through all the child of the given state and select the one with highest UCB value
        """
        circuit = []
        action = self.action
        self.root.action = gate
        child = Node()
        end_state = False
        timestep = 0
        N = 6 #number of max iterations

        # Iterate through the children until the CNOT is operable
        while not end_state:
            timestep += 1
            for i in action:
                child.action = i
                end_state, reward, new_gate, act = self.swap_schedule(child.action, end_state, gate)
                if end_state:
                    self.root.children.append(child)
                    break
                child.reward = reward
                child.n = timestep
                self.root.children.append(child)

            if not end_state:
                gate = new_gate

            # Find the best child
            child = self.select_child(self.root)
            print(f' Best child = {child.action}')
            circuit.append(child.action)

            # Not more than 6 iterations for selection
            if timestep == N:
                break
            if end_state:
                gate = new_gate

        circuit.append(gate)
        print(f' Circuit is {circuit}')
        return circuit, reward

    # function for the result of the simulation
    def expand(self):

        # while non_terminal(node):
        #     node = rollout_policy(node)
        # return result(node)
        pass

    # function for randomly selecting a child node
    def rollout_policy(self):
        # return pick_random(node.children)
        pass

    # function for backpropagation
    def backpropagation(self):
        # if is_root(node) return
        # node.stats = update_stats(node, result)
        # backpropagation(node.parent)
        pass

    def select_child(self, root):

        while bool(root.children):
            children = root.children

            best = None
            best_ucb = inf
            for child in children:
                ucb = self.ucb(child)
                if ucb < best_ucb:
                    best_ucb = ucb
                    best = child
            root = best
        return root

    def mcts(self, gate):
        circuit, _ = self.selection(gate)
        self.expand()
        self.backpropagation()
        return circuit


c = Circuit(4)
s = State()
all = Allocation(c)
con = all.connectivity()
circ = c.get_circuit()

a = Agent(c,s)
for i in circ:
    if not a.schedule_gate(con, i):
        break

m = MCTS(a, c, all)
t = [0, 3, 0]
m.mcts(t)
