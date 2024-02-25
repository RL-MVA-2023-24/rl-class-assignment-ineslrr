import random
import os
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent  # Replace DummyAgent with your agent implementation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent()
    agent.collect_samples(int(1e4))
    gamma = .98
    nb_iter = 100
    nb_actions = fqi_agent.env.action_space.n
    agent.rf_fqi(nb_iter, nb_actions, gamma)
    # agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
