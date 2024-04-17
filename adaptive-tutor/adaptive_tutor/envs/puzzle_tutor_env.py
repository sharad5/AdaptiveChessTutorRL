import gym
from gym.spaces import MultiDiscrete, Discrete, Box, Dict, Sequence, Tuple, Text
from adaptive_tutor.envs.components import Student, PuzzleBank
import numpy as np


class PuzzleTutorEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "themes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
        "elo_ratings": list(range(1000, 2000, 100))
    }

    def __init__(self, 
                 render_mode=None, 
                 beginner_elo_rating=1000, 
                 moving_average_reward_window=20):
        super(PuzzleTutorEnv, self).__init__()

        self.beginner_elo_rating = beginner_elo_rating
        self.moving_average_reward_window = moving_average_reward_window
        
        self.action_space = MultiDiscrete([12, 10])
        self.current_student_level = beginner_elo_rating
        self.student = Student(elo_rating=beginner_elo_rating)
        self.puzzle_bank = PuzzleBank()

        self.observation_space = Dict({
            "puzzle_success_history": Sequence(Tuple((Text(20), Discrete(2), Box(low=1000, high=1900, dtype=np.int32)))),
            "themes_covered": Box(low=0, high=1, shape=(12,), dtype=np.int32),
            "elo_covered": Box(low=0, high=1, shape=(10,), dtype=np.int32),
        })

        # Initial observation state
        self.observation_state = {
            "puzzle_success_history": np.array([]),  # Example initial state
            "themes_covered": np.zeros(12, dtype=np.int32),
            "elo_covered": np.zeros(10, dtype=np.int32),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        # observation_keys_to_relay = ["puzzle_success_history", "themes_covered", "elo_covered"]
        # return {key: self.observation_state[key] for key in self.observation_state.keys()}
        return { key:(value[-1*(self.moving_average_reward_window):] if key=="puzzle_success_history"
          else value) for key, value in self.observation_state.items() }
        # return {}

    def _get_info(self):
        return {"info": None}

    def _set_puzzle(self, elo_rating, theme):
        return self.puzzle_bank.sample_puzzle(elo_rating, theme)
    
    def _student_attempt_puzzle(self, puzzle):
        return self.student.solve_puzzle(puzzle)

    def _compute_reward(self):
        observation = self._get_obs()

        # r1 = MovingAverage(success * ELO Rating of Puzzle, 20)
        relevant_puzzle_history = observation["puzzle_success_history"][(-1*self.moving_average_reward_window):]
        r1 = np.array(list(map(lambda x: x[1]*x[2], relevant_puzzle_history))).mean()

        # r2 = (# Themes Successfully solved)
        r2 = observation["themes_covered"].sum()

        # r3 = (# Puzzles Covered)
        r3 = observation["puzzle_success_history"].shape[0]

        return r1 + r2 - r3

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.observation_state = {
            "puzzle_success_history": np.array([]).reshape(-1,3),  
            "themes_covered": np.zeros(12, dtype=np.int32),
            "elo_covered": np.zeros(10, dtype=np.int32),
        }

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        
        theme_index, puzzle_elo_rating_index = action
        # print(puzzle_elo_rating_index)
        puzzle_elo_rating = self.metadata["elo_ratings"][puzzle_elo_rating_index-1]
        theme = self.metadata["themes"][theme_index-1]
        # print(puzzle_elo_rating, theme)
        sampled_puzzle = self._set_puzzle(puzzle_elo_rating, theme)
        # print(sampled_puzzle)
        puzzle_success = self._student_attempt_puzzle(sampled_puzzle)
        
        # Update the puzzle_success_history in Observational State
        puzzle_success_tuple = np.array([sampled_puzzle["id"], int(puzzle_success), sampled_puzzle["elo_rating"]]).reshape(-1,3)
        self.observation_state["puzzle_success_history"] = np.append(self.observation_state["puzzle_success_history"], puzzle_success_tuple, axis=0)
        
        # Update the themes_covered in Observational State
        theme_index = self.metadata["themes"].index(theme)
        self.observation_state["themes_covered"][theme_index] = 1

        # Update the elo_covered in Observational State
        puzzle_elo_rating_index = self.metadata["elo_ratings"].index(puzzle_elo_rating)
        self.observation_state["elo_covered"][puzzle_elo_rating_index] = 1

        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()
        terminated = self.current_student_level == 1900

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        pass
