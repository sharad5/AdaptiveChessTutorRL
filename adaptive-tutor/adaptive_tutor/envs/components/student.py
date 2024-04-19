import pandas as pd
import chess
import chess.engine

class Student:
    def __init__(self, elo_rating):
        # Initialize student attributes
        self.engine = None
        self.elo_rating = elo_rating # update later

        # TODO: Hardcoding for now, change later
        self.lc0_weights = "/Users/shikharrastogi/AdaptiveChessTutorRL/maia_weights/maia_1100.pb"
        self.depth = 1
        self.time = 0.1

    def initialize_engine(self):
        # Initialize the engine with specified LCZero weights
        self.engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/lc0")
        self.engine.configure({"WeightsFile": self.lc0_weights})

    def close_engine(self):
        # Close the engine
        if self.engine:
            self.engine.quit()

    def predict_move(self, fen, move_start):
        # Setup the board and apply teacher's move
        board = chess.Board(fen)
        board.push(move_start)

        # Get the move predicted by the engine
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth, time=self.time))
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
    
# if __name__=='__main__':
#     student = Student(elo_rating=1200)
#     data = pd.read_csv('/Users/shikharrastogi/Downloads/lichess_db_puzzle.csv')
#     puzzle_row_index = 3764376  # Row index of the puzzle you want to solve
#     row = data.iloc[puzzle_row_index]
#     student.solve_puzzle(row)

'''
Example usage
    
data = pd.read_csv('decompressed_puzzle.csv')
puzzle_row_index = 3764376  # Row index of the puzzle you want to solve
row = data.iloc[puzzle_row_index]

# initialize bot's weight and depth
student = Student(elo_rating=1200, lc0_weights="maia_Weights/maia_1900.pb", depth=7, time=1)

success = student.solve_puzzle(puzzle=row)
print("Success:", success)

# Close the engine when changing bot
student.close_engine()

'''

