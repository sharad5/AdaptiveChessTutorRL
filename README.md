# Adaptive Chess Tutor
Goal: Develop an adaptive teaching system by modeling Teacher as an RL agent. The Students are simulated through heuristics controlled Chess Bots.

Main Github Link: https://github.com/sharad5/AdaptiveChessTutorRL/tree/main

## Environment
Custom RL Environment - `PuzzleTutorEnv-v0`

### Steps to run the Environment
- Register the Environment: `pygmentize adaptive-tutor/adaptive_tutor/__init__.py`
- Create Python package: `pygmentize adaptive-tutor/setup.py`
- Install the python package: `pip install -e adaptive-tutor`

## Agents (Stable-Baselines)

### A2C (Actor-Critic) Agent
- `cd adaptive-tutor`
- `python adaptive_tutor/agents/a2c_sb_agent.py`

### DQN (Deep Q Network) Agent
- `cd adaptive-tutor`
- `python adaptive_tutor/agents/dqn_sb_agent.py`

### A2C Agent additional experiment 
(Branck Link: https://github.com/sharad5/AdaptiveChessTutorRL/tree/a2c_avg)
- `cd adaptive-tutor`
- `python adaptive_tutor/agents/a2c_sb_agent.py`


