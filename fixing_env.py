#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gym
import gym_nmt
from modified_subproc import SubprocVecEnv
import torch.nn as nn
import torch
import numpy as np
from utils import VecPyTorch
import fairseq

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0,'/scratch/msy290/sem2/pytorch-a2c-ppo-acktr')

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
def make_env(env_id, n_missing_words):
    def _thunk():
        env = gym.make(env_id)
        env.init_words(n_missing_words)

        return env

    return _thunk
def reshape_all(obs):
    max_len = 0
    for pair in obs:
        s,t = pair
        max_len = max(max_len,s.shape[1],t.shape[1])
    bigs = []
    bigt = []
    for pair in obs:
        s,t = pair
        news = torch.zeros([s.shape[0],max_len])
        newt = torch.zeros([t.shape[0],max_len])
        news[:,:s.shape[1]] = s
        newt[:,:t.shape[1]] = t
        bigs.append(news)
        bigt.append(newt)
    return (bigs,bigt)


# In[2]:


num_processes = 5
envs = [make_env(env_id = 'nmt-v0',n_missing_words=1)
            for i in range(num_processes)]


# In[3]:


envs = SubprocVecEnv(envs)
envs = VecPyTorch(envs,'cuda')


# In[4]:


n_epochs = 10
training_scheme = [1]*10 + [2]*20 + [3]*20

base_kwargs={'recurrent': False,'dummyenv':envs.dummyenv,'n_proc':num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space,'Attn',base_kwargs)      


# In[5]:


agent = algo.PPO(actor_critic, 0.2, 4, 4,
                         0.5, 0.001, lr=int(7e-4),
                               eps=int(1e-5),
                               max_grad_norm=0.5)


# In[6]:


num_steps = 5
n_epochs = 20
use_gae = False
gamma = 0.99
tau = 0.95
EOS_token = 1
rewards = []


# In[7]:


for epoch in range(n_epochs+1):
    
    n_missing_words = training_scheme[epoch]
    rollouts = RolloutStorage(num_steps*2*(n_missing_words+1)+1, num_processes,
                        envs.observation_space.shape, envs.action_space)
    
    obs = []
    masks = []
    for step in range(num_steps):
        
        ob = envs.reset()
        obs.append(ob)
#         obs = reshape_all()
        
        
#         rollouts.obs[0].copy_(torch.stack(obs).permute(1,0,2))
        
        for n in range(2*n_missing_words+1):

            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(ob)
            
            if (n == 2*n_missing_words):
                action = (torch.ones([num_processes,1])*2).cuda()
            ob, reward, done, infos = envs.step(action)
            obs.append(ob)
#           mask = torch.FloatTensor([[0.0] if done else [1.0]])
    ob1 = obs
    obs = reshape_all(obs)
    rollouts.insert(obs, action, action_log_prob, value, torch.tensor(reward))

        
    next_value = 0 #Doubtful

    rollouts.compute_returns(next_value, use_gae, gamma, tau)
    value_loss, action_loss, dist_entropy = agent.update(rollouts)


    rollouts.after_update()


# In[ ]:




