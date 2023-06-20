
import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import IterableDataset
from state import CircuitState, TopologyState, TimeState
import copy

from environment import ReplayBuffer, Environment, Experience
from transformer_decoder import GPT

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
    
class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, 
                 block_size: int = 38,
                 vocab_size: int = 7,
                 n_layer: int = 8,
                 n_head: int = 4,
                 n_embd: int = 64,
                 dropout: float = 0.0,
                 bias: bool = True,
                 output_dim: int=6):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.gpt = GPT(block_size,
                       vocab_size,
                       n_layer,
                       n_head,
                       n_embd,
                       dropout,
                       bias)
        
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(n_embd, output_dim))

    def forward(self, x):
        x = self.gpt(x.long())
        x = self.mlp(x)

        return x
    
class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self,
                 env: Environment,
                 replay_buffer: ReplayBuffer = None,
                 inference: bool = False) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.inference = inference

        if not self.inference:
            self.replay_buffer = replay_buffer

        self.state = self.env.get_first_state()

        self.swap_array = torch.tensor([self.env.swap_array])

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        #print("resetted@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        self.env.reset()

    def get_action(self, model: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            model: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon and not self.inference:
            action = self.env.sample_action()
        else:
            state = torch.tensor(self.state).unsqueeze(0)

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = model(state)
            swap_array = self.swap_array#.to(device)
            #print("swap array: ",swap_array)
            #print("q values: ", q_values)
            _, action = torch.max(q_values*swap_array, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        model: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            model: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(model, epsilon, device)
        #print("random action drawn: ", action)

        # do step in the environment
        new_state, reward, done, next_state = self.env.step(action)
        #print(new_state)
        #print("next state: ", next_state)
        #print(reward)

        exp = Experience(self.state, action, reward, done, new_state)

        if not self.inference:
            self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()

        return reward, done