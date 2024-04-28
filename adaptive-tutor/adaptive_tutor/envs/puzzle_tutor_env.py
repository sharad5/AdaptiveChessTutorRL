import gym
from gym.spaces import MultiDiscrete, Discrete, Box, Dict, Sequence, Tuple, Text
from adaptive_tutor.envs.components import Student, PuzzleBank
import numpy as np


class PuzzleTutorEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "themes": ['checkmate_patterns', 'checkmating_tactics', 'tactical_themes', 'advanced_tactical_themes', 'strategic_concepts', 'pawn_related_themes', 'piece_specific_endgames', 'king_safety_and_attack', 'special_moves', 'defensive_tactics'],
        "elo_ratings": list(range(1000, 2000, 100)),
        "puzzle_rating_brackets": ['gt_1900', '1400-1500', '1200-1300', '1000-1100', '1500-1600',
       '1800-1900', '1100-1200', '1700-1800', 'lt_900', '1600-1700',
       '900-1000', '1300-1400']
    }

    def __init__(self, 
                 render_mode=None, 
                 beginner_elo_rating=1100,
                 moving_average_reward_window=20):
        super(PuzzleTutorEnv, self).__init__()

        self.beginner_elo_rating = beginner_elo_rating
        self.moving_average_reward_window = 1
        
        self.action_space_lst = list(np.load('adaptive_tutor/action_space.npy', allow_pickle=True))
        self.action_space = Discrete(120)
        self.current_student_level = beginner_elo_rating
        self.student = Student(elo_rating=beginner_elo_rating)
        self.puzzle_bank = PuzzleBank()
        #self.learning_rate = 0.5

        # self.observation_space = Dict({
        #     "puzzle_success_history": Sequence(Tuple((Text(20), Discrete(2), Text(15)))),
        #     "themes_covered": Box(low=0, high=1, shape=(10,), dtype=np.int32),
        #     "num_success_themes_covered": Box(low=0, high=1e6, shape=(10,), dtype=np.int32),
        #     "elo_covered": Box(low=0, high=1, shape=(12,), dtype=np.int32),
        # })

        self.observation_space = Dict({
            "themes_covered": Box(low=0, high=1, shape=(10,), dtype=np.int32),
            "num_success_themes_covered": Box(low=0, high=1e6, shape=(10,), dtype=np.int32)
        })

        self.puzzle_success_history = np.array([]).reshape(-1,3)

        # Initial observation state
        self.observation_state = {
            "themes_covered": np.zeros(10, dtype=np.int32),
            "num_success_themes_covered": np.zeros(10, dtype=np.int32),
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
            return 1900
        elif rating_bracket[:2]=='lt':
            return 500
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


        return r1/1900 + np.min(observation["themes_covered"])/1900 - 3
    
    def _check_terminated(self, themes_covered):

        # Updated
        
        '''av = 0
        for val in themes_covered:
            av += val 
        av = av / len(themes_covered)
        if av <= self.current_student_level:
            return False'''

        if self.current_student_level<=1700:
            return False
        '''for val in themes_covered:
            if val<self.current_student_level - 300:
                return False'''

        return True
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.puzzle_success_history = np.array([]).reshape(-1,3)
        self.observation_state = {
            "themes_covered": np.zeros(10, dtype=np.int32),
            "num_success_themes_covered": np.zeros(10, dtype=np.int32)
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

        return observation, info
    
    '''def learning_rule(self, x=5, y=5):
        SOLVED_PUZZLES = x
        SOLVED_THEMES = y
        num_success_themes = np.sum(np.where(self.observation_state.get('themes_covered') > 0, 1, 0))
        success_at_rating = np.where(self.observation_state.get('themes_covered') > self.current_student_level - 100, 1, 0)
        num_success_puzzles_at_rating = np.sum(success_at_rating * self.observation_state.get('num_success_themes_covered'))
        for key, val in self.elo_aggregates.items():
            #print(key)
            #print(self._convert_bracket_to_reward(key))
            if self._convert_bracket_to_reward(key)>self.current_student_level and val['count']>0:
                #if val['sum']/val['count']>0.5 and val['sum']>5 and val['count']>10:
                #if val['sum']>10 and val['count']>20 and :
                if num_success_themes>SOLVED_THEMES and num_success_puzzles_at_rating>SOLVED_PUZZLES:
                    if self.current_student_level < 1500:
                        self.current_student_level +=200
                        self.student.change_bot(200)
                        self.learning_rate = self.learning_rate * 0.9
                        return True
        return False'''
    
    def learning_rule(self, x=5, y=5):
        SOLVED_PUZZLES = x
        SOLVED_THEMES = y

        '''elo_agg = {}
        themes_agg = {}'''
        total_sum_theme = 0
        total_count_theme = 0
        total_sum_elo = 0
        total_count_elo = 0
        '''for elo_bucket in self.metadata["puzzle_rating_brackets"]:
            elo_agg[elo_bucket] = {'sum': 0, 'count': 0}

        for themes_bucket in self.metadata["themes"]:
            themes_agg[themes_bucket] = {'sum': 0, 'count': 0}'''

        if len(self.puzzle_success_history)>50:
            puzzle_history = self.puzzle_success_history[-100:]
        else:
            puzzle_history = self.puzzle_success_history

        for i in puzzle_history:
            theme, success, rating_bracket = i[0], int(i[1]), i[2]
            if self._convert_bracket_to_reward(rating_bracket)>=self.current_student_level and self._convert_bracket_to_reward(rating_bracket)<self.current_student_level+200:
                total_sum_elo += success
                total_count_elo +=1
            if self._convert_bracket_to_reward(rating_bracket)>=self.current_student_level:
                total_sum_theme += success
                total_count_theme += 1
        
        if total_count_elo>0:
            if total_sum_elo>15 and total_sum_theme>5 and total_sum_elo/total_count_elo>0.55:
                self.current_student_level +=200
                self.student.change_bot(200)
                #self.learning_rate = self.learning_rate * 0.9
                return True
        
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
        #print(self.puzzle_success_history)
                
        # Update the themes_covered in Observational State
        theme_index = self.metadata["themes"].index(theme)
        

        if puzzle_success:
            self.observation_state["themes_covered"][theme_index] = (self.observation_state["num_success_themes_covered"][theme_index] * self.observation_state["themes_covered"][theme_index] +  sampled_puzzle['Rating'])/(self.observation_state[ "num_success_themes_covered"][theme_index]+1)
            # self.observation_state["themes_covered"][theme_index] = max(self.observation_state["themes_covered"][theme_index], sampled_puzzle['Rating'])
            self.observation_state[ "num_success_themes_covered"][theme_index] += 1

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
        bot_update = self.learning_rule()
        if bot_update:
            print('Bot_Upgraded')
        # TODO: Change this logic with whatever the final level is
        terminated = self._check_terminated(self.observation_state["themes_covered"])

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        pass
