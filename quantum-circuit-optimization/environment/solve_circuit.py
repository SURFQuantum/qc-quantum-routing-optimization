from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import Tensor, nn

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from agent import DQN, Agent
from environment import Environment
from state import TopologyState, CircuitState
from argparse import ArgumentParser

def load_model(args):
    model = DQN()
    state_dict = torch.load(args.model_path)["state_dict"]
    #print(state_dict)
    state_dict = {k.replace("net.", ""): v for k, v in state_dict.items() if ("net." in k) and not ("target_net." in k)}
    #for k, v in state_dict.items():
    #    print(k) 
    #print(state_dict)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def main(args):
    circuit = [[(0,1), (2,3)], [(0,2)]]
    target_topology = [(0,1),(1,2),(2,3)]

    distance_metric = "floyd-warshall"
    environment = Environment(circuit, target_topology, distance_metric)
    print(environment.circuit)
    model = load_model(args)

    agent = Agent(environment, distance_metric, inference=True)
    done = False
    reward = 0
    count = 0
    while not done:
            reward, done = agent.play_step(model, epsilon=1.0)
            print("Done?", done, "current reward: ", reward)
            print(agent.env.circuit)
            if count > 10:
                break
            count += 1

if __name__=="__main__":
      
    # Create the parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument('--model_path', type=str, default="./logs/lightning_logs/version_96/checkpoints/epoch=199-step=800.ckpt", help='Input model path')

    # Parse the arguments
    args = parser.parse_args()
    main(args)