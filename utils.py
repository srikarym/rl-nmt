import gym
import gym_nmt
import torch
from baselines.common.vec_env import VecEnvWrapper
import torch.nn as nn
import torch


def reshape_batch(obs):
    max_len = 0
    for pair in obs:
        s,t = pair
        max_len = max(max_len,s.shape[1],t.shape[1])
    bigs = []
    bigt = []
    for pair in obs:
        s,t = pair
        news = torch.ones([s.shape[0],max_len])
        newt = torch.ones([t.shape[0],max_len])
        news[:,:s.shape[1]] = s
        newt[:,:t.shape[1]] = t
        bigs.append(news)
        bigt.append(newt)
    return (bigs,bigt)

class VecPyTorch(VecEnvWrapper):
	def __init__(self, venv, device):
		"""Return only every `skip`-th frame"""
		super(VecPyTorch, self).__init__(venv)
		self.device = device
		self.dummyenv = gym.make('nmt-v0')
		self.pad_val = self.dummyenv.task.source_dictionary.pad()
		# TODO: Fix data types

	def pad(self,obs):
	    obs = list(map(list, zip(*obs)))
	    source = obs[0]
	    target = obs[1]
	    source = sorted(source,key = len,reverse=True)
	    target = sorted(target,key = len,reverse=True)
	    m = max(len(source[0]),len(target[0]))
	    sp = nn.utils.rnn.pad_sequence([torch.ones([m])] + [torch.tensor(s) for s in source] ,batch_first=True,padding_value=self.pad_val)

	    tp = nn.utils.rnn.pad_sequence([torch.ones([m])] + [torch.tensor(s) for s in target] ,batch_first=True,padding_value=self.pad_val)
	    return (sp[1:], tp[1:])
	def reset(self):
		
		obs = self.venv.reset()
		return self.pad(obs)


	def step_async(self, actions):
		actions = actions.squeeze(1).cpu().numpy()
		self.venv.step_async(actions)

	def step_wait(self):
		obs, reward, done, info = self.venv.step_wait()
		

		# obs = torch.from_numpy(obs).float().to(self.device)
		reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
		return self.pad(obs), reward, done, info
