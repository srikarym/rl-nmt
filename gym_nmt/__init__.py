from gym.envs.registration import register

register(
    id='nmt_fake-v0',
    entry_point='gym_nmt.envs:NMTEnv_fake',
)

register(
    id='nmt_easy-v0',
    entry_point='gym_nmt.envs:NMTEnvEasy',
)
register(
    id='nmt_easy_2-v0',
    entry_point='gym_nmt.envs:NMTEnvEasy_2',
)
