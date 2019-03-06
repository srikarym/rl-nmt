from gym.envs.registration import register

register(
    id='nmt_train-v0',
    entry_point='gym_nmt.envs:NMTEnv_train',
)

register(
    id='nmt_eval-v0',
    entry_point='gym_nmt.envs:NMTEnv_eval',
)
