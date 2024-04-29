import gym
from gym.spaces import MultiDiscrete, Discrete, Box, Dict, Sequence, Tuple, Text
from adaptive_tutor.envs.components import Student, PuzzleBank
import numpy as np
from collections import deque
import heapq
import random

class PuzzleTutorEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "themes": ['checkmate_patterns', 'tactical_themes', 'strategic_concepts', 'pawn_related_themes', 'piece_specific_endgames'],
        "elo_ratings": list(range(1000, 2000, 100)),
        "puzzle_rating_brackets": ['1000-1100', '1100-1200', '1200-1300', '1300-1400', '1400-1500',
       '1500-1600', '1600-1700', 'gt1700']
    }

    def __init__(self, 
                 render_mode=None, 
                 beginner_elo_rating=1100,
                 moving_average_reward_window=20):
        super(PuzzleTutorEnv, self).__init__()

        self.beginner_elo_rating = beginner_elo_rating
        self.moving_average_reward_window = 1
        
        self.action_space_lst = list(np.load('adaptive_tutor/action_space_small.npy', allow_pickle=True))
        self.action_space = Discrete(40)
        self.current_student_level = beginner_elo_rating
        self.student = Student(elo_rating=beginner_elo_rating)
        self.puzzle_bank = PuzzleBank()
        self.learning_rate = 0.5
        self.puzzle_windows = [FixedMaxHeap() for _ in range(5)]
        self.lr = None

        # self.observation_space = Dict({
        #     "puzzle_success_history": Sequence(Tuple((Text(20), Discrete(2), Text(15)))),
        #     "themes_covered": Box(low=0, high=1, shape=(10,), dtype=np.int32),
        #     "num_success_themes_covered": Box(low=0, high=1e6, shape=(10,), dtype=np.int32),
        #     "elo_covered": Box(low=0, high=1, shape=(12,), dtype=np.int32),
        # })

        self.observation_space = Dict({
            "themes_covered": Box(low=0, high=1, shape=(5,), dtype=np.int32),
            "num_success_themes_covered": Box(low=0, high=1e6, shape=(5,), dtype=np.int32)
        })

        self.puzzle_success_history = np.array([]).reshape(-1,3)

        # Initial observation state
        self.observation_state = {
            "themes_covered": np.zeros(5, dtype=np.int32),
            "num_success_themes_covered": np.zeros(5, dtype=np.int32),
        }

        # Elo Aggregates & Success Rates
        self.elo_aggregates = {}
        self.elo_buckets_success_rate = {}

        for elo_bucket in self.metadata["puzzle_rating_brackets"]:
            self.elo_aggregates[elo_bucket] = {'sum': 0, 'count': 0}
            self.elo_buckets_success_rate[elo_bucket] = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        # observation_keys_to_relay = ["puzzle_success_history", "themes_covered", "elo_covered"]
        # return {key: self.observation_state[key] for key in self.observation_state.keys()}

        # return { key:(value[-1*(self.moving_average_reward_window):] if key=="puzzle_success_history"
        #   else value) for key, value in self.observation_state.items() }
        return self.observation_state

        # return {}

    def _get_info(self):
        return {"info": self.elo_buckets_success_rate}

    def _set_puzzle(self, action):
        return self.puzzle_bank.sample_puzzle(action)
    
    def _student_attempt_puzzle(self, puzzle):
        return self.student.solve_puzzle(puzzle)
    
    def _convert_bracket_to_reward(self, rating_bracket):
        if rating_bracket[:2]=='gt':
            return 1700
        elif rating_bracket[:2]=='lt':
            return 900
        else:
            return int(rating_bracket.split('-')[0])

    def _compute_reward(self):
        observation = self._get_obs()

        # r1 = MovingAverage(success * ELO Rating of Puzzle, 20)
        relevant_puzzle_history = self.puzzle_success_history[(-1*self.moving_average_reward_window):]
        r1 = np.mean([int(x[1]) * self._convert_bracket_to_reward(x[2]) for x in relevant_puzzle_history])

        # r2 = (# Themes Successfully solved)
        r2 = observation["themes_covered"].sum()

        # r3 = (# Puzzles Covered)
        r3 = self.puzzle_success_history.shape[0]

        

        return -1 + ((np.abs(r1 - self.current_student_level)<=100) * r1)/1900
    
    def _check_terminated(self, themes_covered):

        # Updated
        
        '''av = 0
        for val in themes_covered:
            av += val 
        av = av / len(themes_covered)
        if av <= self.current_student_level:
            return False'''

        return self.current_student_level>=1700
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.puzzle_success_history = np.array([]).reshape(-1,3)
        self.observation_state = {
            "themes_covered": np.zeros(5, dtype=np.int32),
            "num_success_themes_covered": np.zeros(5, dtype=np.int32)
        }

        for elo_bucket in self.metadata["puzzle_rating_brackets"]:
            self.elo_aggregates[elo_bucket] = {'sum': 0, 'count': 0}
            self.elo_buckets_success_rate[elo_bucket] = 0
        observation = self._get_obs()
        info = self._get_info()
        self.current_student_level = self.beginner_elo_rating
        self.student = Student(elo_rating=self.beginner_elo_rating)
        # if self.render_mode == "human":
        #     self._render_frame()
        self.puzzle_windows = [FixedMaxHeap() for _ in range(5)]

        #Setting a learning rule randomly at start of episode
        x = random.choice([100])
        y = random.choice([3])

        print("Starting episode with x,y =", x,  y)
        self.lr = LearningRule(x, y)

        return observation, info
    
    def update_bot(self):
        if self.lr.learning_rule(self.current_student_level, self.observation_state.get('themes_covered')):
            self.student.change_bot(200)
            self.current_student_level+=200
            return True
        else:
            return False



    def step(self, action_idx):
        # Action will be a number from 0-119 inclusive
        action = self.action_space_lst[action_idx]
        
        rating_bracket, theme = action
        sampled_puzzle = self._set_puzzle(action)
        # print(sampled_puzzle)
        puzzle_success = self._student_attempt_puzzle(sampled_puzzle)
        
        
        # Update the puzzle_success_history in Observational State
        puzzle_success_tuple = np.array([theme, int(puzzle_success), rating_bracket]).reshape(-1,3)
        self.puzzle_success_history = np.append(self.puzzle_success_history, puzzle_success_tuple, axis=0)
        
        # Update the themes_covered in Observational State
        theme_index = self.metadata["themes"].index(theme)
        

        self.puzzle_windows[theme_index].push(puzzle_success * sampled_puzzle['Rating'])

        
        self.observation_state["themes_covered"][theme_index] = self.puzzle_windows[theme_index].get_average()

        # if puzzle_success:
        #     self.observation_state["themes_covered"][theme_index] = sum([val for val in self.puzzle_windows[theme_index]])/len()
        #     self.observation_state["themes_covered"][theme_index] = max(self.observation_state["themes_covered"][theme_index], sampled_puzzle['Rating'])
        self.observation_state[ "num_success_themes_covered"][theme_index] += puzzle_success

        # Elo Bucket Success Rate
        self.elo_aggregates[rating_bracket]['sum'] += puzzle_success
        self.elo_aggregates[rating_bracket]['count'] += 1
        self.elo_buckets_success_rate[rating_bracket] =  self.elo_aggregates[rating_bracket]['sum'] /  self.elo_aggregates[rating_bracket]['count']

        #################################################
        #print(self.elo_aggregates)
        #print(self.elo_buckets_success_rate)
        #print(self.observation_state["themes_covered"])
        

        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()
        #print(puzzle_success_tuple)
        bot_update = self.update_bot()
        if bot_update:
            print('Bot_Upgraded')
        terminated = self._check_terminated(self.observation_state["themes_covered"]) or len(self.puzzle_success_history)>=500

        if self.render_mode == "human":
            self._render_frame()

        if terminated:
            print("Number of steps", len(self.puzzle_success_history))
            # reward += (100 - len(self.puzzle_success_history))
        return observation, reward, terminated, False, info

    def render(self, mode="human"):
        return self._get_obs()

    def close(self):
        pass

class FixedMaxHeap:
    def __init__(self):
        self.heap = []
        self.size = 0

    def push(self, num):
        if self.size < 5:
            heapq.heappush(self.heap, num)
            self.size += 1
        elif num > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, num)

    def get_average(self):
        return sum(self.heap)/self.size

class LearningRule:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def learning_rule(self, curr_elo_rating, obs_state):

        elo_cutoff = curr_elo_rating - self.x #X
        num_theme_cutoff = self.y #Y

        # min_elo_solved = np.min(self.observation_state.get('themes_covered'))

        # num_success_themes = np.sum(np.where(self.observation_state.get('themes_covered') > 0, 1, 0))
        # print('num_success_themes', num_success_themes)

        return np.sum(obs_state > elo_cutoff) >= num_theme_cutoff
