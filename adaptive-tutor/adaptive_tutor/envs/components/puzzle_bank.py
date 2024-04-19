import pandas as pd
import random

class PuzzleBank:
    def __init__(self):
        # Initialize a list of puzzles
        self.lichess_data = pd.read_csv('adaptive_tutor/lichess_db_puzzle.csv')
        self.bank = pd.read_pickle('adaptive_tutor/puzzle_bank.pkl')

    def _get_puzzle(self, puzzleId):
        return self.lichess_data.loc[self.lichess_data.PuzzleId==puzzleId].squeeze()

    def sample_puzzle(self, action):
        '''
        action: (rating_bracket, theme) tuple
        '''
        puzzleId = random.choice(self.bank.loc[action].PuzzleId)
        return self._get_puzzle(puzzleId)