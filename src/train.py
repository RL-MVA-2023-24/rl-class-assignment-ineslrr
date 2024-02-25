from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
      self.env= TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
      self.S = []
      self.A = []
      self.R = []
      self.S2 = []
      self.D = []
      self.Qfunctions = []
      self.Q = None
      
    def collect_samples(self, nb_samples, disable_tqdm=False, print_done_states=False):
      s, _ = self.env.reset()
      #dataset = []
      
      for _ in tqdm(range(nb_samples), disable=disable_tqdm):
          a = self.env.action_space.sample()
          s2, r, done, trunc, _ = self.env.step(a)
          #dataset.append((s,a,r,s2,done,trunc))
          self.S.append(s)
          self.A.append(a)
          self.R.append(r)
          self.S2.append(s2)
          self.D.append(done)
          if done or trunc:
              s, _ = self.env.reset()
              if done and print_done_states:
                  print("done!")
          else:
              s = s2
      self.S = np.array(self.S)
      self.A = np.array(self.A).reshape((-1,1))
      self.R = np.array(self.R)
      self.S2= np.array(self.S2)
      self.D = np.array(self.D)
  

    def rf_fqi(self, iterations, nb_actions, gamma, disable_tqdm=False):
      nb_samples = self.S.shape[0]
      
      self.SA = np.append(self.S,self.A,axis=1)
      for iter in tqdm(range(iterations), disable=disable_tqdm):
          if iter==0:
              value=self.R.copy()
          else:
              Q2 = np.zeros((nb_samples,nb_actions))
              for a2 in range(nb_actions):
                  A2 = a2*np.ones((self.S.shape[0],1))
                  S2A2 = np.append(self.S2,A2,axis=1)
                  Q2[:,a2] =  self.Qfunctions[-1].predict(S2A2)
              max_Q2 = np.max(Q2,axis=1)
              value = self.R + gamma*(1-self.D)*max_Q2
          Q = RandomForestRegressor()
          Q.fit(self.SA,value)
          self.Qfunctions.append(Q)
      self.Q=self.Qfunctions[-1]
      self.save()
                    
    def greedy_action(self, Q,s,nb_actions):
      Qsa = []
      for a in range(nb_actions):
          sa = np.append(s,a).reshape(1, -1)
          Qsa.append(Q.predict(sa))
      return np.argmax(Qsa)

    def plot_inital_state(self, nb_iter):
      s0,_ = self.env.reset()
      Vs0 = np.zeros(nb_iter)
      for i in range(nb_iter):
          Qs0a = []
          for a in range(self.env.action_space.n):
              s0a = np.append(s0,a).reshape(1, -1)
              Qs0a.append(self.Qfunctions[i].predict(s0a))
          Vs0[i] = np.max(Qs0a)
      plt.title('Value of inital state across iterations')
      plt.xlabel("Iterations")
      plt.ylabel("Value")
      plt.plot(Vs0)

    def plot_Bellman_residual(self, nb_iter):
      # Bellman residual
      residual = []
      for i in range(1,nb_iter):
          residual.append(np.mean((self.Qfunctions[i].predict(self.SA)-self.Qfunctions[i-1].predict(self.SA))**2))
      plt.figure()
      plt.title("Bellman residual")
      plt.xlabel("Iterations")
      plt.ylabel("Residual")
      plt.plot(residual);

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        if use_random:
          action=self.env.action_space.sample()
        else:
          action=self.greedy_action(self.Q, observation, 1)
        return action

    def save(self, path: str=None) -> None:
        # filename = "my_model.pickle"
        if path is None:
            path = 'Q_tree.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)
            print("Q saved!")

    def load(self) -> None:
        ### SAVE IN THE REPOSITORY
        path = 'Q_tree.pkl'
        if not os.path.exists(path):
            print("No model to load")
            return
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)
            print("Q loaded!")
  
    
