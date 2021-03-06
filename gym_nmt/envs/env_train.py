import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding
import torch
from sacrebleu import sentence_bleu 
# eos_token = 1
import os
import time
import random
import fairseq
from fairseq import tasks
import itertools
from copy import deepcopy


class NMTEnv_train(gym.Env):
	metadata = {'render.modes': ['human']}

	max_len = 100

	def __init__(self):
		self.previous = None
		self.source = None
		self.target = None
		self.action = None
		self.observation = np.ones((2, self.max_len))
		self.missing_target = None
		self.gt = None

	def myinit(self, n_missing_words,train_data,task):
		self.task = task
		self.train_data = train_data
		self.n_vocab = len(task.target_dictionary)
		self.action = spaces.Discrete(self.n_vocab)
		self.n_missing_words_old = n_missing_words

	def seed(self, seed=None): #Don't know how this works
		self.np_random, seed = seeding.np_random(seed)
		return [seed]



	def step(self, action):     #Returns [source, all the previously generated tokens], reward, is_over, true actions

		action,tf = action
		# action = action.squeeze(1).cpu().numpy()
		self.take_action(action,tf)

		reward = self.get_reward(action)

		ob = [self.source,self.previous]
		is_over = self.is_done(action)

		gt = self.get_gt()

		new_word = False

		return np.array(ob), reward, is_over, (gt,new_word)

	def transition(self,nwords):
		self.n_missing_words_old += nwords

	def get_gt(self): #Returns true action for calculating rank
		if (self.steps_done>=len(self.missing_target)):
			gt = self.task.target_dictionary.pad()
		else:
			gt = self.missing_target[self.steps_done]
		return gt

	def is_done(self,action):     
		if action == self.task.target_dictionary.eos() or len(self.generation) == self.n_missing_words+1:

			return True
		return False

	def reset(self):

		self.n_missing_words = self.n_missing_words_old
		training_pair = random.sample(self.train_data,1)[0]

		self.source = training_pair['net_input']['src_tokens'].numpy().tolist()[0]
		self.target = training_pair['target'].numpy().tolist()[0]
		self.generation = []
		self.steps_done = 0
		self.max_reward = 0
		self.index = training_pair['id']
		self.previous = training_pair['net_input']['prev_output_tokens'].numpy().tolist()[0]

		if self.n_missing_words > len(self.previous) - 1:
			self.n_missing_words = len(self.previous) -1
			# self.previous = [self.previous[0]] 

		self.previous = self.previous[:-1*self.n_missing_words]

		self.missing_target = deepcopy(self.target[-1*self.n_missing_words-1:]) 

		new_word = True
		return np.array([self.source,self.previous]),(self.missing_target[0],new_word)  


	def _render(self, mode='human', close=False):
		pass

	def take_action(self,action,tf):
		if tf:
			self.previous.append(self.get_gt())
		else:
			self.previous.append(int(action))
		self.generation.append(int(action))
		self.steps_done = self.steps_done+1
			

	def get_reward(self,action):
		
		reward = 0

		# if (self.steps_done <= self.n_missing_words + 1):
		# 	if int(action) == self.missing_target[self.steps_done-1]:

		# 		reward = self.max_reward + 1/(self.n_missing_words + 1)
		# 		self.max_reward = reward

		# 	else:
		# 		reward = self.max_reward

		ref = self.task.target_dictionary.string(torch.tensor(self.missing_target))
		hyp_incl = self.task.target_dictionary.string(torch.tensor(self.generation))

		if len(self.generation) != 1:
			hyp_excl = self.task.target_dictionary.string(torch.tensor(self.generation[:-1]))

		reward_incl = sentence_bleu(hyp_incl,ref)
		reward_excl = sentence_bleu(hyp_excl,ref) if len(self.generation) != 1 else 0

		reward = (reward_incl - reward_excl)/100

		# if self.is_done(action):

		# 	ref = self.task.target_dictionary.string(torch.tensor(self.missing_target))
		# 	hyp = self.task.target_dictionary.string(torch.tensor(self.generation))

		# 	reward = sentence_bleu(hyp,ref)/100

		return reward


	@property
	def action_space(self):
		return self.action
	@property
	def observation_space(self):
		return self.observation

