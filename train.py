import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gym
import gym_nmt
from modified_subproc import SubprocVecEnv
from utils import VecPyTorch
import time
import numpy as np
import torch

import sys
sys.path.insert(0,'/scratch/msy290/sem2/gym_nmt/pytorch-a2c-ppo-acktr')

from a2c_ppo_acktr import algo
# from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from arguments import get_args
from utils import reshape_batch

args = get_args()


def make_env(env_id, n_missing_words):
    def _thunk():
        env = gym.make(env_id)
        env.init_words(n_missing_words)

        return env

    return _thunk



training_scheme = []


for i in range(args.max_missing_words):
    training_scheme.extend([i]*args.n_epochs_per_word)


envs = [make_env(env_id = 'nmt-v0',n_missing_words=training_scheme[0])
            for i in range(args.num_processes)]

envs = SubprocVecEnv(envs)
envs = VecPyTorch(envs,'cuda')


base_kwargs={'recurrent': False,'dummyenv':envs.dummyenv,'n_proc':args.num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space,'Attn',base_kwargs)


agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, 30,
                 args.value_loss_coef, args.entropy_coef, lr=args.lr,
                       eps=args.eps,
                       max_grad_norm=args.max_grad_norm)


len_train_data = len(envs.dummyenv.data)
sen_per_epoch = len_train_data//(args.num_steps*args.num_processes)

rollouts = RolloutStorage(args.num_steps*2*training_scheme[0]+2, args.num_processes,
                        envs.observation_space.shape, envs.action_space)

for epoch in range(args.n_epochs+1):

    n_missing_words = training_scheme[epoch]


    if (epoch%args.n_epochs_per_word == 0 and epoch!=0):

        envs = [make_env(env_id = 'nmt-v0',n_missing_words=n_missing_words)
            for i in range(args.num_processes)]
        envs = SubprocVecEnv(envs)
        envs = VecPyTorch(envs,'cuda')

        rollouts = RolloutStorage(args.num_steps*(2*n_missing_words+1)+1, args.num_processes,
                        envs.observation_space.shape, envs.action_space)


    obs = []
    rewards = []

    for step in range(args.num_steps):

        ob = envs.reset()
        obs.append(ob)


        for n in range(2*n_missing_words+1):

            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(ob)

            if (n == 2*n_missing_words):
                action = (torch.ones([args.num_processes,1])*2).cuda()
            ob, reward, done, infos = envs.step(action)
            obs.append(ob)

            rollouts.insert( action, action_log_prob, value, torch.tensor(reward))

    rollouts.insert_obs(reshape_batch(obs))

    next_value = actor_critic.get_value(ob)

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
    value_loss, action_loss = agent.update(rollouts)

    rollouts.after_update()
