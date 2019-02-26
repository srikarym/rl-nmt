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


class NMTEnv_fake(gym.Env):
	metadata = {'render.modes': ['human']}

	max_len = 100

	def __init__(self):
		self.previous = None
		self.source = None
		self.target = None
		self.action = None
		self.observation = np.ones((2, self.max_len))
		self.missing_target = None
		self.tac = None

	def init_words(self, n_missing_words,train_data,task):
		self.task = task
		self.train_data = train_data
		self.n_missing_words = n_missing_words
		self.n_vocab = len(task.target_dictionary)
		self.action = spaces.Discrete(self.n_vocab)

	def seed(self, seed=None): #Don't know how this works
		self.np_random, seed = seeding.np_random(seed)
		return [seed]



	def step(self, action):     #Returns [source, all the previously generated tokens], reward, episode_over, true actions

		self.take_action(action)

		reward = self.get_reward(action)

		ob = [self.source,self.previous]
		episode_over = self.is_done(action)

		tac = self.get_tac()
		return np.array(ob), reward, episode_over, (tac,self.index)

	def get_tac(self): #Returns true action for calculating rank
		if (self.steps_done>=len(self.missing_target)):
			tac = self.task.target_dictionary.eos()
		else:
			tac = self.missing_target[self.steps_done]
		return tac

	def is_done(self,action):     
		if action == self.task.target_dictionary.eos() or len(self.generation) == 2*self.n_missing_words+1:
			return True
		return False

	def reset(self):

		
		training_pair = random.sample(self.train_data,1)[0]

		self.source = training_pair['net_input']['src_tokens'].numpy().tolist()[0]
		self.target = training_pair['target'].numpy().tolist()[0]
		self.generation = []
		self.missing_target = deepcopy(self.target[-1*self.n_missing_words-1:]) #should be -1 to include fullstop
		self.steps_done = 0
		self.index = training_pair['id']

		if len(self.target) - 1 <= self.n_missing_words:
			self.previous = [self.task.target_dictionary.eos()]
		else:
			self.previous = training_pair['net_input']['prev_output_tokens'].numpy().tolist()[0][:-1*self.n_missing_words] #remove -1 
		return np.array([self.source,self.previous]),(self.missing_target[self.steps_done],self.index)  


	def _render(self, mode='human', close=False):
		pass

	def take_action(self,action):
		# print('action to take is',action)
		self.previous.append(self.get_tac())
		self.generation.append(int(action))
		self.steps_done = self.steps_done+1
			

	def get_reward(self,action):
		
		reward = 0

		if (self.steps_done <= self.n_missing_words + 1):
			if self.generation == self.missing_target[:self.steps_done]:
				reward = len(self.generation)/(self.n_missing_words + 1)

		# if (self.is_done(action)):

		# 	if self.generation == self.missing_target:
		# 		reward = 1

		return reward


	@property
	def action_space(self):
		return self.action
	@property
	def observation_space(self):
		return self.observation

