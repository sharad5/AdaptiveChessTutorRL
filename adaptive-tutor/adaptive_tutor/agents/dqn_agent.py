import numpy as np
import gym

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__


from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import adaptive_tutor

env = gym.make('adaptive_tutor/PuzzleTutorEnv-v0', render_mode=None) 

def preprocess_observation(obs):
    # Flatten the dictionary into a single array
    # print(obs)
    if isinstance(obs, dict):
        state = obs
    if isinstance(obs, tuple):
        state, info = obs
    # print(state)
    themes_covered = state['themes_covered'].astype(np.float32)
    num_success_themes_covered = state['num_success_themes_covered'].astype(np.float32)
    return np.concatenate([themes_covered, num_success_themes_covered])

nb_actions = 120#env.action_space.n
obs_shape = preprocess_observation(env.reset()).shape

print(f"Observation Space: {obs_shape}")

input_layer = Input(shape=(1,)+obs_shape)
flattened_layer = Flatten()(input_layer)
hidden1 = Dense(24, activation='relu')(flattened_layer)
hidden2 = Dense(24, activation='relu')(hidden1)
output_layer = Dense(nb_actions, activation='linear')(hidden2)
model = Model(inputs=input_layer, outputs=output_layer)

# model = Sequential()
# model.add(Input((1,)+obs_shape))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(nb_actions, activation='linear'))

# model.add(Flatten(nb_actions, activation='linear'))
print(model.summary())

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

class CustomProcessor(Processor):
    def process_observation(self, observation):
        # print(f"In process_observation. Observation: {observation}")
        return preprocess_observation(observation)

    def process_state_batch(self, batch):
        # print(f"In process_state_batch. Observation: {[observation for observation in batch]}")
        return batch#np.array([preprocess_observation(observation) for observation in batch])

    def process_reward(self, reward):
        return reward

    def process_action(self, action):
        return action
    
    def process_info(self, info):
        return {"info": 0}

dqn.processor = CustomProcessor()

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_puzzle_tutor_weights.h5f', overwrite=True)

# Evaluate our algorithm for a few episodes.
dqn.test(env, nb_episodes=5, visualize=True)