from __future__ import division

import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


import argparse
import numpy as np
import pandas as pd
import gym
import itertools
import tqdm

from time import time
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt


# hyperparams
INPUT_SHAPE = (4,)
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 1000000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 0.00025
GAMMA = 0.9
EPSILON_DECREASE_RATE = 0.00001
MAX_EPSILON = 0.3
MIN_EPSILON = 0.005
UPDATE_FREQ = 4
TARGET_NET_UPDATE_FREQ = 100

# gym environment name
#env_name = "Breakout-ram-v0"
env_name = "CartPole-v1"

# saving path
weights_filename = "models/model_1.h5"

# display visualization or not
render = True

prewarming = True

# display logs when training
# 0: no logging information
# 1: info on episode ends
# 2: info on every step
verbosity = 1

# average line when plotting the graph
avg_line_update_rate = 10

class DQNRelpayer():
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index = range(capacity),
            columns = ['observation', 'action', 'reward',
                        'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self,size):
        indices = np.random.choice(self.count, size = size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DQNAgent():
    """
    an agent that uses double Q learning
    but first let us use Q learning
    """
    def __init__(self, env, gamma = 0.99, start_eps = 1, end_eps = 0.001,
                replayer_capacity = 1000000, batch_size = 64, update_freq = 4,
                target_update_freq = 10000, **kwargs):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.epsilon = start_eps
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        
        self.replayer = DQNRelpayer(replayer_capacity)
        self.env = env

        # getting from kwargs
        self.replay_start_size = kwargs.get("replay_start_size", 50000)
        self.clip_rewards = kwargs.get("clip_rewards", False)
        self.epsilon_decrease_rate = kwargs.get("epsilon_decrease_rate", 0.0001)

        # keep track of which step we are
        self.step = 0
        self.fit_count = 0

        # create models
        self.evalutate_net = self.build_net()
        self.target_net = self.build_net()

        # load model
        if os.path.isfile(weights_filename):
            print('loading weights from file:' + weights_filename)
            self.evalutate_net.load_weights(weights_filename)
            self.target_net.load_weights(weights_filename)
        
        # percentage bar
        self.pbar = tqdm.tqdm(total = self.replay_start_size)
        self.pbar.set_description("prewarming... ")


    def build_net(self, callbacks = None):
        model = Sequential([
            Dense(32, input_shape = INPUT_SHAPE, activation = 'relu'),
            Dense(32, activation = 'relu'),
            Dense(self.action_n, activation='linear')
        ])
        print(model.summary())

        optimizer = Adam(lr=.0005)
        if not callbacks:
            callbacks = self.get_callbacks()
        model.compile(loss = 'mse', optimizer = optimizer)
        return model

    
    def decide(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_n)
        else:
            q_values = self.evalutate_net.predict(state)
            action = np.argmax(q_values)
        return action

    
    def learn(self, state, action, reward, next_state, done):
        # store the experience (several frames)
        self.replayer.store(state, action, reward, next_state, done)
        self.step += 1

        # prewarm percentage bar
        if self.replayer.count <= self.replay_start_size:
            self.pbar.update(1)
            if self.replayer.count == self.replay_start_size:
                self.pbar.close()
                global prewarming
                prewarming = False
                print("prewarm done.")

        # does not learn until a few steps or if we do not have enough sample in memory
        if self.step % self.update_freq == 0 and not prewarming:
            # experience replay
            # for some reason I find it cannot be used in for loop directly
            # so I did a few prepocessing
            states, actions, rewards, next_states, dones = self.replayer.sample(self.batch_size)
            if self.clip_rewards:
                reward = np.clip(reward, -1, 1)

            next_qs = self.target_net.predict(next_states)
            next_max_ps = next_qs.max(axis = -1)

            target_qs = self.evalutate_net.predict(states)
            target_qs[range(self.batch_size), actions] = \
                rewards + self.gamma * next_max_ps * (1 - dones)
            #predictions = self.evalutate_net.predict(states)
            #max_qs = predictions.max(axis = -1)
            #predictions[range(self.batch_size), actions] = rewards + self.gamma * max_qs * (1 - dones)
            #self.evalutate_net.fit(states, predictions, verbose = 0, epochs = 1)
            self.evalutate_net.fit(states, target_qs, verbose = 0)
            self.fit_count += 1

            if self.fit_count % self.target_update_freq == 0:
                #transfer the weights of evalutate net to target net
                if verbosity >= 2:
                    print("*" * 30)
                    print("updating target network...")
                    print("*" * 30)

                self.target_net.set_weights(self.evalutate_net.get_weights())
                
            # annealing epsilon
            if self.step > self.replay_start_size:
                self.epsilon = max(self.epsilon - self.epsilon_decrease_rate, self.end_eps)
            
            if verbosity >= 2:
                print("step: %s, reward: %s, epsilon: %s" %(self.step, reward, round(self.epsilon, 4)))

    def get_callbacks(self):
        # tensorboard
        # type tensorboard --logdir=log/    to visualize
        tensorboard = TensorBoard(log_dir = 'logs/{}'.format(time()))
        return [tensorboard,]

    
    def save(self):
        self.evalutate_net.save(weights_filename)


if __name__ == "__main__":
    env = gym.make(env_name)
    agent = DQNAgent(env,GAMMA,start_eps=MAX_EPSILON,end_eps=MIN_EPSILON,replayer_capacity=REPLAY_MEMORY_SIZE,
            batch_size=BATCH_SIZE, update_freq=UPDATE_FREQ,target_update_freq=TARGET_NET_UPDATE_FREQ,
            replay_start_size = REPLAY_START_SIZE)
    
    history_episode_rewards = []
    average_episode_rewards = []
    max_epsiode_reward = -999999
    try:
        for episode in itertools.count():
            state = env.reset()
            epsiode_reward = 0

            for step in itertools.count():
                if render:
                    env.render()
                # agent take action
                action = agent.decide(state[np.newaxis, :])
                observation, reward, done, _ = env.step(action)
                epsiode_reward += reward
                agent.learn(state, action, reward, observation, done)
                state = observation

                if done:
                    if epsiode_reward > max_epsiode_reward and not prewarming:
                        if verbosity >= 1:
                            print('*' * 40)
                            print('saving model of episode reward %s' %epsiode_reward)
                            print('*' * 40)
                        agent.save()    # save model on episode ends
                        max_epsiode_reward = epsiode_reward
                    break
            if not prewarming:
                history_episode_rewards.append(epsiode_reward)
                if episode % avg_line_update_rate == 0:
                    average_episode_rewards.append(sum(history_episode_rewards) / episode)
            if verbosity >= 1 and not prewarming:
                print("episode: %s, reward: %s" %(episode, epsiode_reward))

    except KeyboardInterrupt:
        print("=" * 40)
        print("training session complete...")
        print("=" * 40)
        # display graph
        plt.figure(figsize=(8,4))
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.plot(range(len(history_episode_rewards)),history_episode_rewards)
        plt.plot(np.array(range(len(average_episode_rewards))) * avg_line_update_rate, average_episode_rewards)
        plt.show()

        is_to_save = input("save weights? (y/n)")
        if is_to_save in ('Y', 'y'):
            print("saving weights...")
            agent.save()
            print("save complete.")
        else:
            print("weights discarded.")




