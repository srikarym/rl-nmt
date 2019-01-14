from gym.envs.registration import register

register(
    id='nmt-v0',
    entry_point='gym_nmt.envs:NMTEnv',
)
