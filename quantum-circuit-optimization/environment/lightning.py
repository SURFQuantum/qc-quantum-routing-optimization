from collections import OrderedDict
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from agent import DQN, RLDataset, Agent
from environment import ReplayBuffer, Environment
from state import TopologyState, CircuitState

class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 128,
        lr: float = 1e-2,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 100,
        eps_last_frame: int = 1000,
        eps_start: float = 0.6,
        eps_end: float = 0.01,
        episode_length: int = 32,
        warm_start_steps: int = 10000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.net = DQN()

        self.target_net = DQN()

        #create and environment
        # TODO: simulate many environments
        circuit = [[(0,1), (2,3)], [(0,2)]]
        target_topology = [(0,1),(1,2),(2,3)]

        distance_metric = "floyd-warshall"
        self.env = Environment(circuit, target_topology, distance_metric)

        self.replay_buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, inference=False)
        self.max_time = 16
        
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        print("Populating the buffer...")
        #print("first one!!:",self.agent.env.circuit)
        time_step = 1
        for pass_through in range(100):
            print()
            print("PASSSS THORUGH [===============================]", pass_through)
            self.agent = Agent(self.env, inference=False)
            print("original !!:",self.agent.env.circuit)
            for i in range(self.agent.env.original_circuit_depth):
                exp = self.agent.play_step(self.net, epsilon=1.0)
                self.replay_buffer.append(exp)
                print("after swap: ", i, self.agent.env.circuit)
                print("length of circuit?", self.agent.env.circuit.length())
                print("current timestep in circuit", self.agent.env.current_time_step)

                if exp.done or self.max_time==self.agent.env.circuit.length():
                    print("WE DONE RESETTING@@@@@")
                    self.agent.reset()
                    time_step = 1
                    self.agent.reset_time(time_step)
                    break

                if i==self.agent.env.original_circuit_depth-1:
                    print("RESETTING TIME%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                    time_step += 1
                    self.agent.reset_time(time_step)
                    self.agent.env.steps_of_time = time_step
                    break
            if len(self.replay_buffer)>=steps:
                break
                
        raise ValueError()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        actions = actions.long()

        #print("states in the batch", states)
        #print("states in the next states", next_states)

        state_action_values = self.net(states)#.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        #print("RAW ACTION VALUES", state_action_values)
        #print("ACTIONS@@@@@", actions)
        state_action_values = state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        #print("RAW ACTION VALUES", state_action_values)

        with torch.no_grad():
            device = self.get_device(batch)
            swap_array = torch.tensor(self.env.swap_array).unsqueeze(0).to(device).float()
            next_state_values = self.target_net(next_states)
            #print("ajhlkdgflsjkdf", swap_array*next_state_values)

            next_state_values = torch.max(swap_array*next_state_values, dim=1)[0]
            #print("NEXT STATE ACTION SWAP MULTIPLY", next_state_values)
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        #print("expected:", expected_state_action_values.dtype)

        #print("exptected state action!!!!!!!", expected_state_action_values)
        #print("next state values!!!!#########,", state_action_values)

        #print("LOSSSSSSSSSSSSS", nn.MSELoss()(state_action_values, expected_state_action_values)/10)

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        #print("epsilon value: ", epsilon)
        self.log("epsilon", epsilon)

        # step through environment with agent
        exp = self.agent.play_step(self.net, epsilon, device)
        self.replay_buffer.append(exp)
        reward, done = exp.reward, exp.done
        #print("reward: ", type(reward))
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.replay_buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size, num_workers=0, pin_memory=True
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

model = DQNLightning()

trainer = Trainer(  # limiting got iPython runs
    max_epochs=200,
    val_check_interval=50,
    logger=CSVLogger(save_dir="logs/"),
)

trainer.fit(model)

#print(model.env.circuit)

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
#display(metrics.dropna(axis=1, how="all").head())
sn.relplot(data=metrics, kind="line")
plt.show()