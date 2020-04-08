import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

# Change length of training
EPISODE_COUNT = 25_000

# Hyperparameters for Q-learning
DISCOUNT_FACTOR = 0.95
ALPHA = 0.1  # Learning Rate
INIT_RANGE = [-2, 0]
BIN_SIZE = 40
EPSILON = 0.0  # Inital Exploration Rate
EPSILON_DECAY_START = 1
EPSILON_DECAY_END = EPISODE_COUNT // 2

# Record values
SHOW_EVERY = 500
STATS_EVERY = 100


def random_episode():
    """
    Get to know the enviroment better by printing the fields and going through one episode where the agent
    does random actions.
    """
    env = gym.make('MountainCar-v0')

    # Action space is [0, 1, 2] -> [left, none, right] acceleration
    print("Action Space:", env.action_space)
    # Observation space is [position, velocity]
    print("Observation Space:", env.observation_space)

    # Specific Env Parameters
    print(f"Range of Positions: [{env.min_position}, {env.max_position}]")
    print(f"Max Speed:", env.max_speed)
    print(f"Goal Position and Velocity: [{env.goal_position}, {env.goal_velocity}]")
    # Note: we can pass goal_velocity as a parameter to constructor

    observation, info = env.reset()
    episode_reward = 0

    done = False
    while not done:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        # observation is [position, velocity]
        # reward is -1 for every step and 0 for flag
        # done at flag or exceeds time limit
        # print(observation, reward, done, info)

        # print(observation)
        # print(cont_to_discrete(observation, env))

        episode_reward += reward

    print("Episode Reward:", episode_reward)
    env.close()


def decay_epsilon(epsilon):
    return epsilon - EPSILON / (EPSILON_DECAY_END - EPSILON_DECAY_START)


def init_q_table(env):
    size = [BIN_SIZE] * len(env.observation_space.low) + [env.action_space.n]
    return np.random.uniform(INIT_RANGE[0], INIT_RANGE[1], size)


def cont_to_discrete(observation, env):
    bin_width = (env.observation_space.high - env.observation_space.low) / np.array([BIN_SIZE] * len(env.observation_space.low))
    result = (observation - env.observation_space.low) / bin_width
    return tuple(result.astype(np.int))


def run_episode(env, q_table, epsilon, show=False):
    episode_reward = 0
    new_state = env.reset()

    done = False
    while not done:

        # Calculate next action
        old_state = cont_to_discrete(new_state, env)
        action = np.argmax(q_table[old_state])
        if np.random.random() <= epsilon: # Go exploring if random number less than epsilon
            action = np.random.randint(0, env.action_space.n)

        # render enviroment, take action
        if show: env.render()
        new_state, reward, done, info = env.step(action)

        # update old Q value
        next_max_Q = np.amax(q_table[cont_to_discrete(new_state, env)])
        if not done:
            q_table[old_state][action] = (1 - ALPHA) * q_table[old_state][action] \
                                         + ALPHA * (reward + DISCOUNT_FACTOR * next_max_Q)
        if done and new_state[0] > env.goal_position:
            q_table[old_state][action] = 0

        # add reward and break if done
        episode_reward += reward

    return episode_reward


def q_table_learning(qtable_folder=None, stats_file=None):
    env = gym.make('MountainCar-v0')
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    q_table = init_q_table(env)
    epsilon = EPSILON

    # train agent
    for episode in range(EPISODE_COUNT + 1):
        ep_rewards.append(run_episode(env, q_table, epsilon, episode % SHOW_EVERY == 0))
        if EPSILON_DECAY_START <= episode <= EPSILON_DECAY_END: epsilon = decay_epsilon(epsilon)
        if episode % STATS_EVERY == 0 and episode > 0:
            average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

            if qtable_folder: np.save(qtable_folder+f"/{episode}-qtable.npy", q_table)

    env.close()

    # save stats
    if stats_file: pickle.dump(ep_rewards, open(stats_file, 'wb'))

    # plot data
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    # Get familiarized with enviroment
    # random_episode()

    # Use q table to solve problem
    q_table_learning(qtable_folder=f'./saves/mountain-car/qtables', stats_file=f'./saves/mountain-car/{time.time()}-stats.p')
