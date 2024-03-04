import gymnasium
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
#from tqdm import tqdm
from DQN import DQNAgent
from FQI import FQIAgent
from DQN2 import DQN2Agent
import pickle
import os
import numpy as np
import torch
import torch.nn as nn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.agent_type="dqn2"
        #self.agent_type="dqn"
        if self.agent_type=="dqn":
            ### BASE CONFIGURATION ################
            config = {'nb_actions':4,
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
          ### TRY OTHER CONFIGURATION #####################
            config = {'nb_actions':4,
          'learning_rate': 0.001,
          'gamma': 0.85,
          'buffer_size': 20_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20_000,
          'epsilon_delay_decay': 500,
          'batch_size': 512,
          'gradient_steps': 3,
          'update_target_strategy': 'ema', # or 'replace'
          'update_target_freq': 600,
          'update_target_tau': 0.001,
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
        
        elif self.agent_type=="fqi":
            self.agent=FQIAgent()
        elif self.agent_type=="dqn2":
            self.agent=DQN2Agent()
        else:
            print("Specify agent type")

    def train(self):
        if self.agent_type=="dqn":
            #max_episode=300
            max_episode=4000
            self.agent.train(max_episode)
        elif self.agent=="fqi":
            self.agent.collect_samples(int(1e4))
            self.agent.rf_fqi()
        else:
            print("Specify agent type")

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        action=self.agent.act(observation, use_random)
        return action

    def save(self, path: str=None) -> None:
        self.agent.save(path)
    
    def load(self) -> None:
        self.agent.load()
  
    
