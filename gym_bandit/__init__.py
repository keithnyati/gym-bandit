from gym.envs.registration import register

register(
    id='bandit-v0',
    entry_point='gym_bandit.envs:BanditEnv',
)