from gym.envs.registration import register

register(
    id="adaptive_tutor/PuzzleTutorEnv-v0",
    entry_point="adaptive_tutor.envs:PuzzleTutorEnv",
)
