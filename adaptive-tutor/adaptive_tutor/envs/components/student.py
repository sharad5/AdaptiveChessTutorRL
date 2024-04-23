import pandas as pd
import chess
import chess.engine

class Student:
    def __init__(self, elo_rating):
        # Initialize student attributes
        self.elo_rating = elo_rating 
        self.lc0_weights = f"maia_Weights/maia_{elo_rating}.pb"
        #print(self.lc0_weights)
        self.engine = None
        weights_depth = {
            '1100' : '1',
            '1300' : '2',
            '1500' : '3',
            '1700' : '4',
            '1900' : '5'
            }
        
        self.depth = weights_depth.get(str(elo_rating), None)
        #print(self.depth)

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


# Example usage

'''
data = pd.read_csv('decompressed_puzzle.csv')
puzzle_row_index = 10  # Row index of the puzzle you want to solve
row = data.iloc[puzzle_row_index]



# initialize bot's weight and depth
student = Student(elo_rating=1100)

success = student.solve_puzzle(puzzle=row)
print("Success:", success)

# Close the engine when changing bot
student.close_engine()
'''
