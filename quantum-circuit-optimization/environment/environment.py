import os
import numpy as np
import copy

from typing import List, Tuple

from state import CircuitState, TopologyState, TimeState
from collections import namedtuple, deque

# Named tuple for storing experience steps gathered during training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.
       FROM: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/reinforce-learning-DQN.html

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
        states

        return (
            np.array(states),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )

class Environment:

    def __init__(self, 
                 circuit: CircuitState, 
                 target_topology: TopologyState,
                 distance_metric: str,
                 max_time: int=32,
                 steps_of_time: int = 1):

        self.current_time_step = 0
        self.original_topology = target_topology
        self.original_circuit = circuit
        self.steps_of_time = steps_of_time

        #print(self.original_circuit)
        self.topology = TopologyState(target_topology)
        self.circuit = CircuitState(circuit)

        self.original_circuit_depth = self.circuit.length()
        self.max_time = max_time

        #print(self.circuit)
        #raise ValueError()

        self.num_qubits = self.topology.num_qubits
        self.distance_metric = distance_metric
        self.allowed_swaps = self.get_swaps(self.topology)
        print("allowed swaps: ", self.allowed_swaps)
        
        # input representation special tokens
        self.S = self.num_qubits
        self.T = self.num_qubits+1
        self.P = self.num_qubits+2

        # actions is a list of tuples
        # action_to_idx: (int, int) -> int
        self.actions, self.action_to_idx = self.action_space()
        #print("action space: ", self.actions)
        #print("num of actions total: ", len(self.actions))
        self.num_actions = len(self.actions)

        # reverse of the above: int -> (int, int)
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

        tuples = list(map(tuple, self.allowed_swaps))

        # the allowed swaps but indexed
        # get the swap indices, a swap (qubit 1, qubit 2) == (qubit 2, qubit 1)
        self.swap_indices = []

        for (q1,q2) in tuples:
            if (q1, q2) in self.action_to_idx:
                self.swap_indices.append(self.action_to_idx[(q1,q2)])
            elif (q2, q1) in self.action_to_idx:
                self.swap_indices.append(self.action_to_idx[(q2, q1)])
        
        self.swap_array = np.zeros(len(self.actions))
        self.swap_array[self.swap_indices] = 1  

        self.first_state = self.get_first_state()

    def reset_time(self, value: int):
        self.current_time_step = 0
        self.steps_of_time = value

    def reset(self):
        self.topology = TopologyState(self.original_topology)
        self.circuit = CircuitState(self.original_circuit)
        self.current_time_step = 0
        self.original_circuit_depth = self.circuit.length()

    def sample_action(self) -> int:
        idx = np.random.choice(len(self.swap_indices))
        return self.swap_indices[idx]

    def get_swaps(self, topology: TopologyState, unique_swaps: bool=False) -> np.ndarray:
        """
        These are the swaps that we can do according to the target topology.
        """
        topology = topology.adjacency_topology
        if unique_swaps:
            # unique swaps means half of the adjacencies are irrelevant
            topology = np.tril(topology)

        qubits_to_swap = np.where(topology==1)
        qubits_to_swap = np.stack(qubits_to_swap).T
        return qubits_to_swap

    def action_space(self) -> Tuple[List, dict]:
        #num_swaps_possible = np.math.factorial(self.num_qubits) / (2*np.math.factorial(self.num_qubits-2))
        #print("num swaps possible: ", num_swaps_possible)
        actions = []
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
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

        #reward = 1 / np.log(distance+1.5)
        #reward = 1 / distance
        reward = -distance

        if distance==0:
            reward = 20
        return reward
    
    def is_terminated(self) -> bool:

        if self.distance_metric=="floyd-warshall":
            distance = self.topology.floyd_warshall_distance_to_circuit(self.circuit.topology.nx_topology)

        if distance==0:
            return True
      
        #if self.is_truncated():
        #    return 2
        
        return False
    
    def is_truncated(self) -> bool:
        return self.circuit.length() > self.max_time
        #return self.circuit.length() > self.max_time

    # TODO: DOESN'T BELONG IN THIS CLASS
    def get_max_action(self, output_state_actions: np.ndarray) -> Tuple[int, int]:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """       
        # array with 1s where a swap is allowed: e.g.: [0 0 0 1 0 0 1 0 1 1 0]
        
        action = np.argmax(output_state_actions * self.swap_array)

        return self.idx_to_action[action]
    
    def perform_swap(self, swap_qubits: Tuple[int, int]) -> np.ndarray:

        # We are converting from a networkx graph to numpy and then back to nx
        # TODO: find a way to make this more efficient?
        circuit_topology = self.circuit.topology.adjacency_topology
        q1, q2 = swap_qubits

        old_topology = circuit_topology.copy()
        
        # swap rows
        circuit_topology[q1, :], circuit_topology[q2, :] = old_topology[q2, :], old_topology[q1, :]
        old_topology = circuit_topology.copy()

        # swap columns
        circuit_topology[:, q1], circuit_topology[:, q2] = old_topology[:, q2], old_topology[:, q1]

        return circuit_topology
    
    def update_circuit(self, swap_qubits: Tuple[int, int]) -> None:
        # perform the swap by swapping in the adjacency matrix
        new_circuit_topology = self.perform_swap(swap_qubits)

        # update the circuit topology according to the swap
        self.circuit.topology.update(new_circuit_topology)

        # if we are still handling the original circuit timesteps
        if self.current_time_step < self.original_circuit_depth-1:
            self.circuit.insert_circuit(self.current_time_step+self.steps_of_time, [swap_qubits], "SWAP")
            self.circuit.update_circuit_after_swap(swap_qubits, self.current_time_step+self.steps_of_time)

        else:
            self.circuit.add_to_cirq([[swap_qubits]], "SWAP")
    
    def get_first_state(self):
        return self.get_DQN_input(self.circuit.circuit[0], self.circuit.topology)

    def first_step(self, action: int) -> Tuple:
        self.update_circuit(self.idx_to_action[action])
        self.current_time_step += self.steps_of_time

        current_state = self.first_state

        next_state = self.get_DQN_input(self.circuit.circuit[self.current_time_step], self.circuit.topology)

        return current_state, next_state

    def step(self, action: int) -> Tuple:
        """
        action is a swap
        """
        print("self.idx_to_action[action]", self.idx_to_action[action])

        # TODO: we need a way for the agent to not do anything


        if self.current_time_step==0:
           current_state, next_state = self.first_step(action)
        
        else:
            
            current_state = self.get_DQN_input(self.circuit.circuit[self.current_time_step], self.circuit.topology)

            # update the topology by performing the swap 
            # and update the circuit by adding the timestep with the swap
            self.update_circuit(self.idx_to_action[action])
            
            # action is an integer, both ways is valid e.g., (1,2)==(2,1)
            #action = self.action_to_idx[action] if action in self.action_to_idx else self.action_to_idx[(action[1], action[0])]
            # update time step
            self.current_time_step += self.steps_of_time
            print("self.steps_of_time***************************", self.steps_of_time)
            print("self.steps_of_time in circuit length***************************", self.current_time_step)
            next_state = self.get_DQN_input(self.circuit.circuit[self.current_time_step], self.circuit.topology)

        reward = self.get_reward()
        done = self.is_terminated()

        # s, r, d, s'
        return current_state, reward, done, next_state
    
    def get_DQN_input(self, 
                      state: TimeState = None,
                      circuit_topology: TopologyState = None):

        """
        TODO: check the validty of below statement.

        The input to the transformer is a sequence of current arity-2 gates in a specific timestep
        together with the current topology of the circuit. 

        The input is a fixed sequence-length vector of tokens, each corresponding to a qubit. 

        For example: 

        1. max sequence length is 16
        2. we have 4 qubits: [0,1,2,3]
        3. we are at timestep 3, where we have 1 CNOT between (0,1) and (2,3).
        4. Our current circuit topology is: (0,1),(1,2),(2,3)

        The current input vector would look like:
        x = [S, 0, 1, 2, 3, T, 0, 1, 1, 2, 2, 3, P, P, P, P], 
        
                where   S indicates the start of state 
                        T indicates the start of the Topology
                        P indicates padding
        
        Note that we can have a maximum of Q // 2 arity-2 gates in one timestep, and a maximum
        of (Q^2-Q) circuit topology pairs (in the extreme case that all qubits are connected to all others)

        Meaning that our vector should preferably be of length |x| = 2 + (Q // 2)*2 + (Q^2)*2
        So for a 32-qubit circuit we would have an input length of |x| = 2 + 32 + 2048 = 2082

        Here we already see that the topology sequence is dominant. Especially if it should be padded to the fullest.
        Hence, we probably need to make some educated choices.

        Can we just slash (Q^2)*2 by 4? So Q^2 / 2, how big are topologies usually?
        """

        gates = np.array(state.gates).flatten()
        topology = self.get_swaps(circuit_topology, unique_swaps=True).flatten()
        
        gates_length = gates.shape[0]
        topology_length = topology.shape[0]

        x = np.zeros((2+self.num_qubits+(self.num_qubits**2)*2,))
        x = np.zeros((12))
        x[0] = self.S
        #print(x)
        x[self.num_qubits+1] = self.T
        #print(x)
        x[1:1+gates_length] = gates
        #print(x)
        x[1+gates_length:self.num_qubits+1] = self.P
        #print(x)
        x[self.num_qubits+2:self.num_qubits+2+topology_length] = topology
        #print(x)
        x[self.num_qubits+2+topology_length:] = self.P
        #print(x)

        return x

def main():

    """
    TESTING
    """
    circuit = [[(0,1), (2,3)], [(0,2)]]
    
    target_topology = TopologyState([(0,1),(1,2),(2,3)])
    circuit = CircuitState(circuit)

    print("Circuit before update step:",circuit)
    env = Environment(circuit, target_topology, "floyd-warshall")
    env.circuit.topology.draw()
    print("Circuit adjacency topology BEFORE swap\n", env.circuit.topology.adjacency_topology)
    current_state, reward, done, next_state = env.step((3,2))
    print("state vector: ", current_state)

    print("Circuit adjacency topology AFTER swap\n", env.circuit.topology.adjacency_topology)

    env.circuit.topology.draw()
    print("after first step:",circuit)
    current_state, reward, done, next_state = env.step((0,1))
    print(current_state)

    print("after second step:",circuit)

if __name__=="__main__":
    main()