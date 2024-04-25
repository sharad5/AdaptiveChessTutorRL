import pandas as pd
import chess
import chess.engine

class Student:
    def __init__(self, elo_rating):
        # Initialize student attributes
        self.elo_rating = elo_rating 
        self.lc0_weights = f"maia_Weights/maia_{elo_rating}.pb"
        print(self.lc0_weights)
        self.engine = None
        weights_depth = {
            '1100' : '1',
            '1300' : '2',
            '1500' : '3',
            '1700' : '4',
            '1900' : '5'
            }
        
        self.depth = weights_depth.get(str(elo_rating), None)
        print(self.depth)

    def initialize_engine(self):
        # Initialize the engine with specified LCZero weights
        self.engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/lc0")
        self.engine.configure({"WeightsFile": self.lc0_weights})

    def close_engine(self):
        # Close the engine
        if self.engine:
            self.engine.quit()
        self.engine = None

    def predict_move(self, fen, move_start):
        # Setup the board and apply teacher's move
        board = chess.Board(fen)
        board.push(move_start)

        # Get the move predicted by the engine
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        predicted_move = result.move

        return predicted_move

    def calculate_success(self, predicted_move, move_student):
        success = (predicted_move == move_student)
        return success

    def solve_puzzle(self, puzzle):
        # Extract puzzle data
        fen = puzzle['FEN']
        move_start = chess.Move.from_uci(puzzle['Moves'].split(' ')[0])
        move_student = chess.Move.from_uci(puzzle['Moves'].split(' ')[1])

        # Initialize the engine if not already initialized
        if not self.engine:
            self.initialize_engine()

        # Predict the move
        predicted_move = self.predict_move(fen, move_start)

        success = self.calculate_success(predicted_move, move_student)

        return success
    
    # Learning rule: Successfully solves X puzzles around the current bot's level and covers Y themes
    def change_bot(self, change):
        self.close_engine()
        new_elo = self.elo_rating + change
        self.elo_rating = new_elo
        new_lc0_weights = f"maia_Weights/maia_{new_elo}.pb"
        self.lc0_weights = new_lc0_weights
        print(self.lc0_weights)
        
        self.engine = None
        
        weights_depth = {
            '1100' : '1',
            '1300' : '2',
            '1500' : '3',
            '1700' : '4',
            '1900' : '5'
            }
        
        self.depth = weights_depth.get(str(new_elo), None)
        print(self.depth)

# ------------------------------ Example of Student usage ------------------------------ #
'''
    
data = pd.read_csv('decompressed_puzzle.csv')
puzzle_row_index = 28 # Row index of the puzzle you want to solve
row = data.iloc[puzzle_row_index]

# initialize bot's weight and depth
student = Student(elo_rating=1700)

success = student.solve_puzzle(puzzle=row)
print("Success:", success)

# Close the engine when changing bot
student.close_engine()
'''

# ------------------------------ Example of Update usage ------------------------------ #
'''
metadata = {
        "render_modes": ["human"],
        "themes": ['checkmate_patterns', 'checkmating_tactics', 
                   'tactical_themes', 'advanced_tactical_themes', 
                   'strategic_concepts', 'pawn_related_themes', 
                   'piece_specific_endgames', 'king_safety_and_attack', 
                   'special_moves', 'defensive_tactics'],
        "elo_ratings": list(range(1000, 2000, 100)),
        "puzzle_rating_brackets": ['gt_1900', '1400-1500', '1200-1300', '1000-1100', '1500-1600', '1800-1900', 
                                   '1100-1200', '1700-1800', 'lt_900', '1600-1700', '900-1000', '1300-1400']
    }

def rating_to_bracket(numeric_rating):
    if numeric_rating < 900:
        return 'lt_900'
    elif numeric_rating > 1900:
        return 'gt_1900'
    else:
        lower_bound = int(numeric_rating / 100) * 100
        upper_bound = lower_bound + 100
        out = str(lower_bound) + '-' + str(upper_bound)
        return out

# observation_state = {
#     "puzzle_success_history": np.array([]).reshape(-1,3),  
#     "themes_covered": np.zeros(10, dtype=np.int32),
#     "num_success_themes_covered": np.zeros(10, dtype=np.int32),
#     "elo_covered": np.zeros(12, dtype=np.int32),
# }

observation_state = {
    "themes_covered": np.zeros(10, dtype=np.int32),
    "num_success_themes_covered": np.zeros(10, dtype=np.int32),
}

import random

for _ in range(20):

    puzzle_row = random.randint(0, len(data) - 1)
    sampled_puzzle = data.iloc[puzzle_row]
    
    rating_bracket = rating_to_bracket(sampled_puzzle['Rating'])
    
    theme_index = np.random.randint(0, len(metadata.get('themes')) - 1)
    theme = metadata.get('themes')[theme_index]
    
    puzzle_success = bool(random.getrandbits(1))
    
#     # Update the puzzle_success_history in Observational State
#     puzzle_success_tuple = np.array([theme, int(puzzle_success), rating_bracket]).reshape(-1,3)
#     observation_state["puzzle_success_history"] = np.append(observation_state["puzzle_success_history"], 
#                                                             puzzle_success_tuple, axis=0)
    
    # Update the themes_covered in Observational State
    if puzzle_success:
        observation_state["themes_covered"][theme_index] = (observation_state["num_success_themes_covered"][theme_index] 
                                                            * observation_state["themes_covered"][theme_index] 
                                                            + sampled_puzzle['Rating']) / (observation_state["num_success_themes_covered"][theme_index]+1)
        
        observation_state[ "num_success_themes_covered"][theme_index] += 1


#     # Update the elo_covered in Observational State
#     puzzle_elo_rating_index = metadata["puzzle_rating_brackets"].index(rating_bracket)
#     observation_state["elo_covered"][puzzle_elo_rating_index] = 1

# Learning rule: Successfully solves X puzzles around the current bot's level and covers Y themes
SOLVED_PUZZLES = 5 #X
SOLVED_THEMES = 5 #Y

curr_elo_rating = student.elo_rating
print('curr_elo_rating', curr_elo_rating)

success_at_rating = np.where(observation_state.get('themes_covered') > curr_elo_rating - 50, 1, 0)
num_success_puzzles_at_rating = np.sum(success_at_rating * observation_state.get('num_success_themes_covered'))
print('num_success_puzzles_at_rating', num_success_puzzles_at_rating)

num_success_themes = np.sum(np.where(observation_state.get('themes_covered') > 0, 1, 0))
print('num_success_themes', num_success_themes)

if num_success_themes > SOLVED_THEMES and num_success_puzzles_at_rating > SOLVED_PUZZLES:
    if curr_elo_rating < 1900:
        student.change_bot(200)


# Learning rule: Successfully solved a puzzle of X level in every theme offered AND successfully solved Y puzzles in each theme offered
curr_elo_rating = student.elo_rating
print('curr_elo_rating', curr_elo_rating)

SOLVED_PUZZLES = curr_elo_rating - 50 #X
SOLVED_THEMES = 2 #Y

min_elo_solved = np.min(observation_state.get('themes_covered')[observation_state.get('themes_covered') != 0])
print('min_elo_solved', min_elo_solved)

num_success_themes = np.sum(np.where(observation_state.get('themes_covered') > 0, 1, 0))
print('num_success_themes', num_success_themes)

if min_elo_solved > SOLVED_PUZZLES and np.mean(num_success_themes) > SOLVED_THEMES:
    if curr_elo_rating < 1900:
        student.change_bot(200)
'''
