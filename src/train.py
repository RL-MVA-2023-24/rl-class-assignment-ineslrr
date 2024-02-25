from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm
from DQN import DQNAgent
from FQI import FQIAgent
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import 

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__():
        self.agent_type="dqn"
        if self.agent_type=="dqn":
            config = {'nb_actions':n_action,
          'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 20_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10_000,
          'epsilon_delay_decay': 400,
          'batch_size': 1024,
          'gradient_steps': 1,
          'update_target_strategy': 'ema', # or 'replace'
          'update_target_freq': 600,
          'update_target_tau': 0.002,
          'criterion': torch.nn.SmoothL1Loss()}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            nb_neurons=24
            state_dim=6
            n_action=4
            DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)
            self.agent=DQNAgent(config, DQN)
        else:
            print("Specify agent type")

    def train(self):
        if self.agent_type=="dqn":
            max_episode=300
            agent.train(self, max_episode)
        else:
            print("Specify agent type")

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        action=self.agent.act(observation, use_random)
        return action

    def save(self, path: str=None) -> None:
        self.agent.save(path)
    
    def load(self) -> None:
        self.agent.load()
  
    
