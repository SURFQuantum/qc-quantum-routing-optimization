from itertools import tee, permutations
from math import log, sqrt, e, inf
import random


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Node:
    def __init__(self):
        self.reward = 0
        self.visits = 0
        self.children: list[Node] = []
        self.circuit_depth = 1
        self.parent = None
        self.expanded = False
        self.action = None
        self.ucb = 0
        self.cnot = None
        self.distance = inf

    def finished(self):
        return self.distance == 0 or self.circuit_depth == MCTS.max_circuit_depth

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


class MCTS:
    max_circuit_depth = 25

    def __init__(self, connectivity, topology):
        self.reward = 0
        self.parent = 0
        self.N = 0
        self.n = 0
        self.n_qubits = 4

        # self.n_qubits = 6
        self.logic = [0, 1, 2, 3]
        self.root = Node()
        self.connectivity = connectivity
        self.topology = topology
        self.node_evaluator = lambda child, montecarlo: None

    # Node is a circuit with gates, parent node should change for every action, child node is the possible action
    # coming from parent node
    @property
    def action(self):
        possible_action = []
        connectivity_set = set(map(tuple, self.connectivity))

        for i in permutations(self.topology, 2):
            if tuple(i) in connectivity_set:
                q = [0, 0, 1]
                q[0], q[1] = i[1], i[0]
                swap_gate = [i[0], i[1], 1]
                if swap_gate not in possible_action and q not in possible_action:
                    possible_action.append(swap_gate)

        return possible_action

    def ucb(self, node_i):
        """
        Upper Confidence Bound for selecting the best child node
        """
        ucb = node_i.reward + sqrt(2) * (
            sqrt(log(node_i.parent.visits + e + (10 ** (-6))) / (node_i.visits + 10 ** (-1))))
        return ucb

    def swap_circuit(self, a, b, gate):
        """
        :param a:
        :param b:
        :param gate:
        :return:
        """
        gate = [b if x == a else a if x == b else x for x in gate]
        self.logic = [b if x == a else a if x == b else x for x in self.logic]
        gate.append(0)
        return gate, self.logic

    def swap_schedule(self, i, end_state, action):
        """
    
        :param i: 
        :param end_state: 
        :param action: 
        :return: 
        """
        a, b, _ = i
        end_distance = float('inf')

        # CNOT-gate
        new_gate = [action[0], action[1]]

        # Swap the nodes and change the CNOT-gate
        new_gate, _ = self.swap_circuit(a, b, new_gate)

        # Calculate the distance to an operable qubit connectivity location
        for i in self.connectivity:
            q0 = abs(new_gate[0] - i[0])
            q1 = abs(new_gate[1] - i[1])
            distance = q0 + q1
            end_distance = min(distance, end_distance)

        # Reward for improving the CNOT
        if end_distance == 0:

            reward = 1
            end_state = True
        else:
            reward = -1

        return end_state, reward, new_gate, end_distance

    def expansion(self, node):
        """
        Create one row of children to this node
        :param node:
        :return:
        """
        for i in self.action:
            child = Node()
            child.action = i
            end_state, reward, new_gate, end_distance = self.swap_schedule(child.action, False, node.cnot)
            child.distance = end_distance
            child.parent = node
            child.ucb = self.ucb(child)
            child.cnot = new_gate
            child.reward = 0
            child.circuit_depth = node.circuit_depth + 1
            node.add_child(child)

    def mcts(self, root):
        """
        
        :param root: 
        :return: 
        """
        self.expansion(root)

        selected_node = root
        for i in range(self.max_circuit_depth):
            # Find the best option to expand
            selected_child = self.select_child(selected_node)

            # Expand the tree (create children) (1 row)
            self.expansion(selected_child)

            # Simulate until end of sim
            heuristic_value = self.simulate(selected_child)

            self.backpropagation(selected_child, heuristic_value)
            #
            # for child in self.root.children:
            #     print(child.action, end=" ")
            #     if child.visits != 0:
            #         print(child.reward / child.visits, end="")
            #     print("")
            # print("")

        best_child = None
        best_value = -2
        for child in root.children:
            if child.reward / child.visits > best_value:
                best_value = child.reward / child.visits
                best_child = child

        return best_child

        # return circuit, child

    def select_child(self, root):
        current_node = root

        while len(current_node.children) > 0:
            best_child = None
            best_score = float('-inf')

            for child in current_node.children:
                if child.visits == 0:
                    best_child = child
                    break
                score = self.ucb(child)

                if score > best_score:
                    best_score = score
                    best_child = child
            current_node = best_child
            # TODO: Add optimization to not do reversing of swap gates immediately

        return current_node

    # function for backpropagation
    def backpropagation(self, leaf_node, reward):
        """
        
        :param leaf_node: 
        :param reward: 
        :return: 
        """
        while leaf_node.parent is not None:
            leaf_node.reward += reward
            leaf_node.visits += 1
            leaf_node = leaf_node.parent

    def simulate(self, root):
        """
        
        :param root: 
        :return: 
        """
        current_node = root
        while not current_node.finished():
            self.expansion(current_node)
            next_node = random.choice(current_node.children)
            current_node.children.clear()
            current_node = next_node

        # The reward is higher (better), if the circuit depth is smaller
        return (1 - (current_node.circuit_depth / self.max_circuit_depth)) * 2 - 1

    def simulate_ai(self, root):
        """
        This function should return the heuristic value for the passed node
        :param root: the node to be evaluated
        :return: the heuristic value (between 1 and -1)
        """
        return 0  # model.forward(root.encoded)

def main():
    connectivity = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
    topology = [0, 1, 2, 3, 4]
    gate = (4, 0, 0)

    m = MCTS(connectivity, topology)

    root = Node()
    root.action = gate
    root.cnot = gate

    while root.distance is not 0:
        root = m.mcts(root)
        print(root.action)


if __name__ == "__main__":
    main()
