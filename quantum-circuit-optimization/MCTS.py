import itertools
from math import log, sqrt, e

# TODO: FIND SHORTEST PATH TO LEAF NODE
from agent import Agent
from circuit import Circuit
from qubit_allocation import Allocation


class MCTS:

    def __init__(self, agent, circuit):
        self.state = agent.scheduled_gates
        # print(f'the state is {self.state}')# a circuit [[0,1,0],[1,0,0]] from first action
        # self.constraints = allocation_class.connectivity()
        self.reward = 0
        # self.action = 0  #
        self.children = 0
        self.parent = 0
        self.N = 0
        self.n = 0
        self.n_qubits = circuit.n_qubits

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
        mean_reward = node_i.reward
        num_parent_visits = node_i.N
        num_child_visits = node_i.n
        ucb = mean_reward + 2 * (sqrt(log(num_parent_visits + e + (10 ** -6)) / (num_child_visits + 10 ** -10)))
        return ucb

    def selection(self, gate):
        # receives iteration
        # choosing child node based on Upper Confidence Bound
        # UCB(node i) = (mean node value) + confidence value sqrt(log num visits parent / num visits of node i)
        """
        Iterate through all the child of the given state and select the one with highest UCB value
        """

        # get parent node and child_nodes is the one with all the calculated actions, (ADD: constraint not two swaps next to each other)

        # hideously programmed, but at this point I couldnt come up with a different solution for 1 action in a state,
        # instead of all the states in the list at the end of the iteration
        child_node = self.state
        action = self.action
        size = len(child_node)
        child_node.append(0)
        for i in action:
            child_node[size] = i
            a, b, _ = i

            if gate[0] > gate[1]:
                distance = gate[0] - gate[1]
            else:
                distance = gate[1] - gate[0]

            new_gate = [gate[0], gate[1]]

            for x in range(len(new_gate)):
                if new_gate[x] == a:
                    new_gate[x] = b
                elif new_gate[x] == b:
                    new_gate[x] = a

            if new_gate[0] > new_gate[1]:
                new_distance = new_gate[0] - new_gate[1]
            else:
                new_distance = new_gate[1] - new_gate[0]

            # Reward for improving the CNOT
            if new_distance < distance:
                reward = 5

            else:
                reward = -1

            print(reward)


            # TODO: find a reward system and calculation with UCB and random selecting perhaps of a branch, research something about that random selecting?
        return 0

    # function for the result of the simulation
    def rollout(self, node):
        # while non_terminal(node):
        #     node = rollout_policy(node)
        # return result(node)
        return

    # function for randomly selecting a child node
    def rollout_policy(self, node):
        # return pick_random(node.children)
        return

    # function for backpropagation
    def backpropagation(self, node, result):
        # if is_root(node) return
        # node.stats = update_stats(node, result)
        # backpropagation(node.parent)
        return

    # function for selecting the best child
    # node with the highest number of visits
    def best_child(self, node):
        # pick
        # child
        # with the highest number of visits
        return


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


c = Circuit(4)
all = Allocation()
con = all.connectivity()
circ = c.get_circuit()

a = Agent()
for i in circ:
    if not a.schedule_gate(con, i):
        break

m = MCTS(a, c)
t = [3, 0, 0]
m.selection(t)
