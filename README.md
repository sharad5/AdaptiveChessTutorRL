# Updates
This repository contains an enhanced version of the main branch aimed at mitigating randomness issues encountered during training. We follow anapproach where for each action instead of considering single puzzle we consider a set of puzzles. The overall success of an action is determined by the success rate of solving the puzzles within its set. If the success rate exceeds 50%, the action is considered successful.

# Adaptive Chess Tutor
Goal: Develop an adaptive teaching system by modeling Teacher as an RL agent. The Students are simulated through heuristics controlled Chess Bots.

## Environment
Custom RL Environment - `PuzzleTutorEnv-v0`

### Steps to run the Environment
- Register the Environment: `pygmentize adaptive-tutor/adaptive_tutor/__init__.py`
- Create Python package: `pygmentize adaptive-tutor/setup.py`
- Install the python package: `pip install -e adaptive-tutor`

## Agents (Stable-Baselines)

### A2C (Actor-Critic) Agent
- cd adaptive-tutor
- python adaptive_tutor/agents/a2c_sb_agent.py

### DQN (Deep Q Network) Agent
- cd adaptive-tutor
- python adaptive_tutor/agents/dqn_sb_agent.py
