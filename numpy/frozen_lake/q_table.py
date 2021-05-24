import gym
import random
import numpy as np

ALPHA = 0.65
DECAY_FACTOR = 0.975
EPISODE_LENGTH = 500
EPSILON = 1
GAMMA = 0.975

class Q_Table():
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.n_action = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        self.epsilon = EPSILON
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4 * np.random.random([self.state_size, self.n_action])

    def get_action(self, state):
        possible_actions = self.q_table[state]
        if random.random() > self.epsilon:
            action = np.argmax(possible_actions)
        else:
            action = np.random.choice(self.n_action)
        return action

    def learn(self, transition):
        state, action, reward, next_state, done = transition
        q_next = np.zeros([self.n_action]) if done else self.q_table[next_state]
        q_target = reward + GAMMA * np.max(q_next)

        error = q_target - self.q_table[state, action]
        self.q_table[state, action] = self.q_table[state, action] + ALPHA * error

        if done:
            self.epsilon *= DECAY_FACTOR

    def act(self):
        total_return = 0
        for episode in range(EPISODE_LENGTH):
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn([state, action, reward, next_state, done])
                state = next_state
                total_return += reward

        print(f'Final return: {total_return}')

    
if __name__ == '__main__':
    q_table = Q_Table()
    q_table.act()