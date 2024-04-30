import gym
from stable_baselines3 import A2C
import adaptive_tutor
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback


env = gym.make('adaptive_tutor/PuzzleTutorEnv-v0', render_mode=None) 

# Define a wrapper class for observation preprocessing
class CustomPreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.student_level = env.current_student_level
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def observation(self, observation):
        state = observation['themes_covered']
        # return np.append(state, self.student_level)
        return state
    

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=2000,
  save_path="./adaptive_tutor/logs/",
  name_prefix="a2c_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

env = CustomPreprocessingWrapper(env)
env.reset()

# model = DQN.load("logs/rl_model_20000_steps")
# model.load_replay_buffer('logs/rl_model_replay_buffer_20000_steps.pkl')
# model.set_env(env)
model = A2C("MlpPolicy", env, device="cpu", verbose=1)

model.learn(total_timesteps=20000, progress_bar=True, callback=checkpoint_callback)
model.save("a2c_sb")