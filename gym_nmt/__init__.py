from gym.envs.registration import register

register(
    id='nmt-v0',
    entry_point='gym_nmt.envs:NMTEnv',
)

register(
    id='nmt_redbleu-v0',
    entry_point='gym_nmt.envs:NMTEnvRed',
)
