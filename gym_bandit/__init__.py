from gym.envs.registration import register

register(
    id='GreedyBandit-v1',
    entry_point='gym_bandit.envs:GreedyBanditEnv',
)

register(
    id='GradientBandit-v0',
    entry_point='gym_bandit.envs:GradientBanditEnv',
)