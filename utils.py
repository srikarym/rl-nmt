import gym
import gym_nmt
import torch
from baselines.common.vec_env import VecEnvWrapper
import torch.nn as nn
import torch
import numpy as np


class VecPyTorch(VecEnvWrapper):
	def __init__(self, venv, device,pad):
		"""Return only every `skip`-th frame"""
		super(VecPyTorch, self).__init__(venv)
		self.device = device
		self.pad_val = pad
		# TODO: Fix data types

	def pad(self,obs):
		obs = list(map(list, zip(*obs)))
		source = obs[0]
		target = obs[1]

		# ms = len(source[0])
		# mt = len(target[0])
		max_size = 100
		
		sp = nn.utils.rnn.pad_sequence([torch.ones([max_size])] + [torch.tensor(s) for s in source] ,batch_first=True,padding_value=self.pad_val)

		tp = nn.utils.rnn.pad_sequence([torch.ones([max_size])] + [torch.tensor(s) for s in target] ,batch_first=True,padding_value=self.pad_val)
		return (sp[1:], tp[1:])

	def reset(self):
		
		obser = self.venv.reset()
		obs = []
		tac = []
		for ob in obser:
			obs.append(ob[0])
			tac.append(ob[1])
		return self.pad(obs),np.array(tac)


	def step_async(self, actions):
		actions = actions.squeeze(1).cpu().numpy()
		self.venv.step_async(actions)

	def step_wait(self):
		obs, reward, done, tac = self.venv.step_wait()

		reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
		return self.pad(obs), reward, done,np.array(tac)
