import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from types import SimpleNamespace


class GymSolver():
    default = {'EPISODE_COUNT': 25_000,
               'DISCOUNT_FACTOR_GAMMA': 0.5,
               'LEARNING_RATE_ALPHA': 0.5,
               'INIT_RANGE': [-1, 1],
               'BIN_SIZE': 10,
               'EPSILON': 0,
               'EPSILON_DECAY_START': 1,
               'EPSILON_DECAY_END': 2,
               'SHOW_EVERY': 500,
               'STATS_EVERY': 100,
               'epsilon_func': None,
               'decay_epsilon': None,
               'goal': None}

    def __init__(self, env_name, **kwargs):
        self.env_name = env_name

        # Assign hyperparameters as fields
        self.__dict__.update(kwargs)
        for k, v in GymSolver.default.items():
            if k not in self.__dict__.keys(): self.__dict__[k] = v
        self.epsilon = self.EPSILON

        self.ep_rewards = []
        self.aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

        print(f"Hyperparameters: {self.__dict__}")

    def cont_to_discrete(self, observation):
        bin_width = (self.env.observation_space.high - self.env.observation_space.low) / np.array(
            [self.BIN_SIZE] * len(self.env.observation_space.low))
        result = (observation - self.env.observation_space.low) / bin_width
        return tuple(result.astype(np.int))


class TableSolver(GymSolver):
    def __init__(self, env_name, **kwargs):
        super(TableSolver, self).__init__(env_name, **kwargs)

    def init_q_table(self):
        size = [self.BIN_SIZE] * len(self.env.observation_space.low) + [self.env.action_space.n]
        return np.random.uniform(self.INIT_RANGE[0], self.INIT_RANGE[1], size)

    def run_episode(self, goal=None, show=False):
        episode_reward = 0
        new_state = self.env.reset()
        q_table = self.q_table

        done = False
        while not done:

            # Calculate next action
            old_state = self.cont_to_discrete(new_state)
            action = np.argmax(q_table[old_state])
            if np.random.random() <= self.epsilon:  # Go exploring if random number less than epsilon
                action = self.env.action_space.sample()

            # render enviroment, take action
            if show: self.env.render()
            new_state, reward, done, info = self.env.step(action)

            # update old Q value
            next_max_Q = np.amax(q_table[self.cont_to_discrete(new_state)])
            if not done:
                q_table[old_state][action] = (1 - self.LEARNING_RATE_ALPHA) * q_table[old_state][action] \
                                             + self.LEARNING_RATE_ALPHA * (
                                                         reward + self.DISCOUNT_FACTOR_GAMMA * next_max_Q)
            if done and goal and goal(self.env, new_state):
                q_table[old_state][action] = 0

            # add reward and break if done
            episode_reward += reward

        return episode_reward

    def q_table_learning(self, qtable_folder=None, stats_file=None):
        self.env = gym.make(self.env_name)
        self.q_table = self.init_q_table()

        # train agent
        for episode in range(self.EPISODE_COUNT + 1):
            self.ep_rewards.append(self.run_episode(show=episode % self.SHOW_EVERY == 0))
            if self.EPSILON_DECAY_START <= episode <= self.EPSILON_DECAY_END:
                if self.decay_epsilon: self.epsilon = self.decay_epsilon(self)
                if self.epsilon_func: self.epsilon = self.epsilon_func(self, episode)
            if episode % self.STATS_EVERY == 0 and episode > 0:
                average_reward = sum(self.ep_rewards[-self.STATS_EVERY:]) / self.STATS_EVERY
                self.aggr_ep_rewards['ep'].append(episode)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['max'].append(max(self.ep_rewards[-self.STATS_EVERY:]))
                self.aggr_ep_rewards['min'].append(min(self.ep_rewards[-self.STATS_EVERY:]))
                print(
                    f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {self.epsilon:>1.2f}')

                if qtable_folder: np.save(qtable_folder + f"/{episode}-qtable.npy", self.q_table)

        self.env.close()

        # save stats
        if stats_file: pickle.dump(self.ep_rewards, open(stats_file, 'wb'))

        # plot data
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['avg'], label="average rewards")
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['max'], label="max rewards")
        plt.plot(self.aggr_ep_rewards['ep'], self.aggr_ep_rewards['min'], label="min rewards")
        plt.legend(loc=4)
        plt.show()

    def render_one_episode(self, q_table_file):
        total_reward = 0
        self.env = gym.make(self.env_name)
        state = self.env.reset()
        q_table = np.load(q_table_file)

        done = False
        while not done:
            self.env.render()

            action = np.argmax(q_table[self.cont_to_discrete(state)])
            state, reward, done, info = self.env.step(action)
            total_reward += reward

        self.env.close()
        print(f"Completed with reward of {reward}.")

if __name__ == "__main__":
    linear_epsilon_decay = lambda self: self.epsilon - self.EPSILON / (self.EPSILON_DECAY_END - self.EPSILON_DECAY_START)

    gs = TableSolver('MountainCar-v0',
                   INIT_RANGE=[-2, 0],
                   DISCOUNT_FACTOR_GAMMA=0.95,
                   LEARNING_RATE_ALPHA=0.1,
                   BIN_SIZE=40,
                   EPSILON=0.5,
                   EPSILON_DECAY_END=12_500,
                   decay_epsilon=linear_epsilon_decay)

    #gs.q_table_learning(qtable_folder='./saves/mountain-car/qtables',
    #                    stats_file=f'./saves/mountain-car/{time.asctime()}-stats.p')

    gs.render_one_episode('./saves/mountain-car/qtables/25000-qtable.npy')
