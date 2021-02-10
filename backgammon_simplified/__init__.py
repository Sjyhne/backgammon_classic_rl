from gym.envs.registration import register

register(
    id='backgammon-v69',
    entry_point='backgammon_simplified.envs:SimplifiedBackgammonEnv',
)