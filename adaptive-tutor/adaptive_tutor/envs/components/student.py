import numpy as np

class Student:
    def __init__(self, elo_rating):
        # Initialize student attributes
        self.elo_rating = elo_rating
        self.bot = None # Update Later

    def solve_puzzle(self, puzzle):
        # TODO: Chitvan's Puzzle Solving Code
        success = np.random.choice([True, False])
        return success