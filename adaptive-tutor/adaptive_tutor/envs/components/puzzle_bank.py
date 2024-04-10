import random

class PuzzleBank:
    def __init__(self):
        # Initialize a list of puzzles
        self.puzzles = [
            {'id': 1, 'elo_rating': 1200, 'theme': 'A', 'content': 'What is 2+2?'},
            {'id': 2, 'elo_rating': 1300, 'theme': 'B', 'content': 'What is H2O?'},
            {'id': 3, 'elo_rating': 1300, 'theme': 'C', 'content': 'What is H2O?'},
            {'id': 4, 'elo_rating': 1300, 'theme': 'D', 'content': 'What is H2O?'},
            {'id': 5, 'elo_rating': 1300, 'theme': 'E', 'content': 'What is H2O?'},
            {'id': 6, 'elo_rating': 1300, 'theme': 'F', 'content': 'What is H2O?'},
            {'id': 7, 'elo_rating': 1300, 'theme': 'G', 'content': 'What is H2O?'},
            {'id': 8, 'elo_rating': 1300, 'theme': 'H', 'content': 'What is H2O?'},
            {'id': 9, 'elo_rating': 1300, 'theme': 'I', 'content': 'What is H2O?'},
            {'id': 10, 'elo_rating': 1300, 'theme': 'J', 'content': 'What is H2O?'},
            {'id': 11, 'elo_rating': 1300, 'theme': 'K', 'content': 'What is H2O?'},
            {'id': 12, 'elo_rating': 1300, 'theme': 'L', 'content': 'What is H2O?'},
        ]

    def sample_puzzle(self, elo_rating, theme):
        # Placeholder logic: randomly select a puzzle.
        filtered_puzzles = [puzzle for puzzle in self.puzzles if puzzle['theme'] == theme]
        if not filtered_puzzles:
            return None
        return random.choice(filtered_puzzles)