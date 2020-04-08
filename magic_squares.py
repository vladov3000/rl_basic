import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from PIL import Image
import cv2
import time
import random
import queue
import pickle

N_EPISODES = 100
MAX_STEPS = 10

WORLD_N = 10
WORLD_M = 15


class GridWorld(gym.Env):
    PLAYER_COLOR = (255, 175, 0)
    MAGIC_SQUARE_COLOR = (255, 255, 255)
    N_MAGIC_PAIRS = 1

    def __init__(self, n, m):
        self.grid = [0 for i in range(n * m)]
        self.n, self.m = n, m

        self.state_space = [i for i in range(self.n * self.m - 1)]
        self.state_space_plus = self.state_space + [self.n * self.m]
        self.actions = [-self.m, self.m, -1, 1]
        self.action_space = spaces.Discrete(len(self.actions))

        self._set_magic_squares()
        self.agent_pos = 0

    def _set_magic_squares(self):
        # Generate 2 * N_MAGIC_SQUARES random numbers
        random_nums = set()
        while len(random_nums) < 2 * GridWorld.N_MAGIC_PAIRS:
            r = random.randint(0, self.n * self.m)
            if r not in random_nums:
                random_nums.add(r)

        # Assign magic squares
        self.magic_squares = {random_nums.pop(): random_nums.pop() for i in range(0, GridWorld.N_MAGIC_PAIRS)}
        for square in self.magic_squares.keys():
            self.grid[square] = 2

    def _is_terminal_state(self, state):
        return state in self.state_space_plus and state not in self.state_space

    def _set_state(self, state):
        self.grid[self.agent_pos] = 0
        self.agent_pos = state
        self.grid[self.agent_pos] = 1

    def _off_grid_move(self, old_state, new_state):
        if new_state not in self.state_space_plus:
            return True
        elif old_state % self.m == 0 and new_state % self.m == self.m - 1:
            return True
        elif old_state % self.m == self.m - 1 and new_state % self.m == 0:
            return True
        return False

    def _get_2D_pos(self, pos):
        return pos // self.m, pos // self.n

    def step(self, action: int):
        new_state = self.agent_pos + self.actions[action]
        if new_state in self.magic_squares.keys():
            new_state = self.magic_squares[new_state]

        reward = -1 if not self._is_terminal_state(new_state) else 0
        if not self._off_grid_move(self.agent_pos, new_state):
            self._set_state(new_state)
            return new_state, reward, self._is_terminal_state(new_state), None
        return self.agent_pos, reward, self._is_terminal_state(self.agent_pos), None

    def reset(self):
        self.agent_pos = 0
        self.grid = [0 for i in range(self.n * self.m)]
        return self.agent_pos

    def render(self, mode='human', close=False):
        if mode == 'human':
            grid_image = np.zeros((self.n, self.m, 3), dtype=np.uint8)
            grid_image[self._get_2D_pos(self.agent_pos)] = GridWorld.PLAYER_COLOR
            for i in self.magic_squares.items():
                for square in i:
                    grid_image[self._get_2D_pos(square)] = GridWorld.MAGIC_SQUARE_COLOR

            img = Image.fromarray(grid_image, 'RGB')
            img = img.resize((50 * self.m, 50 * self.n))
            cv2.imshow("image", np.array(img))
            cv2.waitKey(1)


def train_random(save_file=None):
    env = GridWorld(WORLD_N, WORLD_M)
    episode_rewards = []

    for episode in range(N_EPISODES):
        observation = env.reset()
        episode_reward = 0

        for t in range(MAX_STEPS):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(observation)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        episode_rewards.append(episode_reward)

    if save_file:
        pickle.dump(episode_rewards, open(save_file, 'wb'))
        print(f"Episode rewards saved to {save_file}")

    plt.plot(episode_rewards)
    plt.show()

def play_actions(actions):
    env = GridWorld(WORLD_N, WORLD_M)
    

def plot_data(files):
    for f in files:
        plt.plot(pickle.load(open(f, 'rb')))
    plt.show()


if __name__ == "__main__":
    train_random("./saves/magic-squares/random-episode_rewards.p")

    # plot_data(["./saves/magic-squares/random-episode_rewards.p"])
