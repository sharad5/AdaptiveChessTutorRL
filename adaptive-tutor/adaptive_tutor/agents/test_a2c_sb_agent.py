import gym
from stable_baselines3 import A2C
import adaptive_tutor
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

NUM_EPISODES_FOR_EVAL = 2
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

env = CustomPreprocessingWrapper(env)
env.reset()

# model = DQN.load("logs/rl_model_20000_steps")
# model.load_replay_buffer('logs/rl_model_replay_buffer_20000_steps.pkl')
# model.set_env(env)
model = A2C.load("a2c_sb.zip")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=NUM_EPISODES_FOR_EVAL)
print(f"Average Reward: {mean_reward}, Std Reward: {std_reward}")