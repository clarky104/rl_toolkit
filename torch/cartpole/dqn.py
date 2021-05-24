import gym
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import matplotlib.pyplot as plt
from common.supplement import plot, is_solved

# hyperparameters
ALPHA = 2e-3
BATCH_SIZE = 16
DECAY_FACTOR = 0.995
EPSILON = 1
GAMMA = .99
MAXLEN = 50000
MIN_EPSILON = 0.01
TRAIN_LENGTH = 2000

class MLP(nn.Module):
    def __init__(self, state_size, n_actions):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.q_values = nn.Linear(16, n_actions)

        self.optim = Adam(self.parameters(), lr=ALPHA)
    
    def forward(self, x):
        x = F.relu(self.fc1(torch.FloatTensor(x)))
        x = F.relu(self.fc2(x))
        q_values = self.q_values(x)
        return q_values

class DQN():
    def __init__(self):
        self.epsilon = EPSILON
        self.env = gym.make('CartPole-v1')
        self.n_action = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.network = MLP(self.state_size, self.n_action)
        self.memory = deque(maxlen=MAXLEN)

    def remember(self, transition):
        self.memory.append(transition)

    def get_action(self, state):
        with torch.no_grad():
            if np.random.random() > self.epsilon:
                q_vals = self.network(state)
                action = torch.argmax(q_vals).item()
            else:
                action = np.random.choice(self.n_action)
            return action

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_pred = self.network(state)
            q_target = reward + (1 - done) * GAMMA * self.network(next_state)

            self.network.optim.zero_grad()
            loss = torch.mean((q_target - q_pred) ** 2)
            loss.backward()
            self.network.optim.step()

    def decrement_epsilon(self):
        self.epsilon *= DECAY_FACTOR
        self.epsilon = max(self.epsilon, MIN_EPSILON)

    def act(self):
        total_return = []
        mean_return = []
        solved = False
        for episode in range(TRAIN_LENGTH):
            episodic_return = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episodic_return += reward
                self.remember([state, action, reward, next_state, done])
                self.learn()
                self.decrement_epsilon()
                state = next_state

            total_return.append(episodic_return)
            mean_return.append(np.mean(total_return[-100:]))
            solved = is_solved(mean_return, solved, episode, TRAIN_LENGTH)

        return total_return, mean_return

if __name__ == '__main__':
    dqn = DQN()
    total_return, mean_return = dqn.act()
    plot(total_return, mean_return, 'DQN')