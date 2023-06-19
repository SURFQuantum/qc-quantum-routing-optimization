import os
import numpy as np
import copy

from typing import List, Tuple

from state import CircuitState, TopologyState
from collections import namedtuple, deque

# Named tuple for storing experience steps gathered in training

State = namedtuple(
    "State",
    field_names=["state", "circuit_topology"])

# FROM: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"])

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )

# class State:
#     def __init__(self,
#                  state: TimeState = None,
#                  circuit_topology: TopologyState = None):
        
#         self.state = state
#         self.circuit_topology = circuit_topology


# class Experience:
#    def __init__(self, 
#                 state: State = None, 
#                 action: Tuple[int, int] = None, 
#                 reward: float = None, 
#                 next_state: State=None):
        
#         self.state = state
#         self.action = action
#         self.reward = reward
#         self.next_state = next_state

class Environment:

    def __init__(self, 
                 circuit: CircuitState, 
                 target_topology: TopologyState,
                 distance_metric: str):

        self.current_time_step = 0

        self.circuit = circuit
        self.original_circuit_depth = self.circuit.length()
        self.topology = target_topology
        self.num_qubits = self.topology.num_qubits
        self.distance_metric = distance_metric
        self.allowed_swaps = self.get_allowed_swaps()

        # actions is a list of tuples
        # action_to_idx: (int, int) -> int
        self.actions, self.action_to_idx = self.action_space()
        print("num of actions total: ", len(self.actions))

        # reverse of the above: int -> (int, int)
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

        tuples = list(map(tuple, self.allowed_swaps))
        print("tuples: ", tuples)

        # the allowed swaps but indexed
        print("action to index: ", self.action_to_idx)
        swap_indices = np.array([self.action_to_idx[tuple] for tuple in tuples])
        
        self.swap_array = np.zeros(len(self.actions))[swap_indices] = 1
        

    def get_allowed_swaps(self) -> np.ndarray:
        print("adjacency: ", self.topology.adjacency_topology)
        qubits_to_swap = np.where(self.topology.adjacency_topology==1)
        qubits_to_swap = np.stack(qubits_to_swap).T
        return qubits_to_swap

    def action_space(self) -> Tuple[List, dict]:
        num_swaps_possible = np.math.factorial(self.num_qubits) / (2*np.math.factorial(self.num_qubits-2))
        print("num swaps possible: ", num_swaps_possible)
        num_swaps_possible = int(num_swaps_possible)
        actions = []
        for i in range(num_swaps_possible):
            for j in range(i+1, num_swaps_possible):
                actions.append((i,j))
        
        action_to_idx = {swap: i for i, swap in enumerate(actions)}

        return actions, action_to_idx

    def get_reward(self) -> float:

        if self.distance_metric=="floyd-warshall":
            distance = self.topology.floyd_warshall_distance_to_circuit(self.circuit.topology.nx_topology)
        #TODO: elif

        # TODO: check if this reward is approproate for now I list the following advantages and disadvantages:
        # 1. Emphasize smaller distances
        # 2. However, as the agent gets closer to the target, the rewards may not strictly increase.
        #    can be a problem?
        # 3. If the distance between states is large, the rewards will become extremely small. 
        #    So the agent would be dealing with sparse rewards and potentially never lift off.
        #    Will curriculum learning help this?

        # I think we could fix this by considering the distance of two current states,
        # Essentially measure the gradient of improvement. This would help (3) and (2).
        # Also, keep in mind that in AlphaGo Zero we use +1 and -1 (big and small) reward
        # for finding a solution to the routing problem. 

        reward = 1 / np.log(distance+1.5)

        return reward
    
    def get_DQN_input(self):
        raise NotImplementedError
    
    def is_terminated(self) -> bool:

        if self.distance_metric=="floyd-warshall":
            distance = self.topology.floyd_warshall_distance_to_circuit(self.circuit.topology.nx_topology)

        if distance==0:
            return True
        
        return False

    # TODO: DOESN'T BELONG IN THIS CLASS
    def get_max_action(self, output_state_actions: np.ndarray) -> Tuple[int, int]:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """       
        # array with 1s where a swap is allowed: e.g.: [0 0 0 1 0 0 1 0 1 1 0]
        
        action = np.argmax(output_state_actions * self.swap_array)

        return self.idx_to_action[action]

    
    def is_truncated(self) -> bool:
        return self.circuit.length() > self.max_time
    
    def perform_swap(self, swap_qubits: Tuple[int, int]) -> np.ndarray:

        # We are converting from a networkx graph to numpy and then back to nx
        # TODO: find a way to make this more efficient?
        circuit_topology = self.circuit.topology.adjacency_topology
        circuit_topology[swap_qubits] = circuit_topology[swap_qubits[1], swap_qubits[0]]

        # update the topology
        return circuit_topology
    
    def update_circuit(self, swap_qubits: Tuple[int, int]) -> None:
        # perform the swap by swapping in the adjacency matrix
        new_circuit_topology = self.perform_swap(swap_qubits)

        # update the circuit topology according to the swap
        self.circuit.topology.update(new_circuit_topology)

        # if we are still handling the original circuit timesteps
        if self.current_time_step < self.original_circuit_depth:
            self.circuit.insert_circuit(self.current_time_step+1, [swap_qubits], "SWAP")

        else:
            self.circuit.add_to_cirq([[swap_qubits]], "SWAP")
    
    def step(self, action: Tuple[int, int]) -> Tuple:
        """
        action is a swap
        """
        
        current_state = State(self.circuit.circuit[self.current_time_step], 
                                            self.circuit.topology)
        # get the maximum action

        # update the topology by performing the swap 
        # and update the circuit by adding the timestep with the swap
        self.update_circuit(action)
        
        # update time step
        self.current_time_step += 1

        # We have updated the current time step and topology already so we are
        # effectively in the next state
        next_state = copy.deepcopy(self.circuit).circuit[self.current_time_step]
        next_state = State(next_state, self.circuit.topology)

        reward = self.get_reward()
        done = self.is_terminated()

        # s, a, r, s'
        return current_state, action, done, reward, next_state

def main():
    circuit = [[(0,1), (2,3)], [(0,2)]]
    
    target_topology = TopologyState([[0,1],[1,2],[2,3]])
    circuit = CircuitState(circuit)
    print("before step:",circuit)
    env = Environment(circuit, target_topology, "floyd-warshall")
    env.circuit.topology.draw()
    current_state, action, reward, done, next_state = env.step((2,3))
    print("after step:",circuit)
    print("current state:",current_state.state)
    print("action:",action)
    print("reward:",reward)
    print("done",done)
    print("next state:",next_state.state)

    env.circuit.topology.draw()

if __name__=="__main__":
    main()