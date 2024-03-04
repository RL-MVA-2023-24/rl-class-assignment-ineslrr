from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

import random
import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
import os
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)    

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class DQN2Agent:  
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        config = {
            'nb_actions':env.action_space.n,
            'gamma':0.98,
            'batch_size':800,
            'buffer_size':100000,
            'epsilon_max':1.,
            'epsilon_min':0.01,
            'epsilon_stop':20000,
            'epsilon_delay':100,
            'criterion':torch.nn.SmoothL1Loss(),
            'learning_rate':0.001,
            'gradient_steps':3,
            'update_target_strategy':'replace', #or 'ema'
            'update_target_freq':400,
            'update_target_tau':0.005,
            'nb_neurons':256
        }

        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        buffer_size = config['buffer_size'] 
        self.memory = ReplayBuffer(buffer_size,device)
        self.model = self.dqn_network(config, device)
        self.target_model = deepcopy(self.model).to(device)
        self.epsilon_max = config['epsilon_max'] 
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_stop'] 
        self.epsilon_delay = config['epsilon_delay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = config['criterion']
        lr = config['learning_rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps']
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']


    def act(self, observation, use_random=False):
        if use_random:
            action = env.action_space.sample()
        else: 
            #Greedy action
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
                action = torch.argmax(Q).item()
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        path = "dqn2.pt"
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()
    
    def dqn_network(self, config, device):
        state_dim = env.observation_space.shape[0]
        n_action = config['nb_actions']
        nb_neurons = config['nb_neurons']
        DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),  
                          nn.Linear(nb_neurons, n_action)).to(device)
        return DQN


    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        max_episode = 300
        previous_val = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.act(state, use_random=True)
            else:
                action = self.act(state, use_random=False)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > 100:
                    validation_score = evaluate_HIV(agent=self, nb_episode=1)
                else :
                    validation_score = 0
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", validation score ", '{:.2e}'.format(validation_score),
                      sep='')
                
                if validation_score > previous_val:
                    print("better model")
                    previous_val = validation_score
                    self.save("model.pt")
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return  