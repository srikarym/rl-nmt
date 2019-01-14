import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding
import torch
from sacrebleu import sentence_bleu 
eos_token = 1

class NMTEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self):
		self.previous = None
		self.source = None
		self.target = None


	def pad(self,vec):
		vec = np.pad(vec[:,0],(0,self.max_len-len(vec)), 'constant',constant_values=-1).reshape(self.max_len,1 )
		return vec

	def space_init(self,n_vocab,max_len,index2word):
		# self.action = np.arange(0,n_vocab)
		self.max_len = max_len
		self.action = spaces.Discrete(n_vocab)
		self.observation = np.ones((2,max_len))
		self.index2word = index2word
		# self.observation = spaces.Box(low=0,high = max_len,shape = (10),dtype = np.int64)


	def seed(self, seed=None): #Don't know how this works
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def my_init(self,source,target,n_missing_words): #call this when an iteration for (source,target) begins
		self.source = source
		self.target = target
		self.n_missing_words = n_missing_words
		self.previous = None

	def step(self, action):     #Returns [source, all the previously generated tokens], reward, episode_over, {}

		self.take_action(action)

		reward = self.get_reward(action)
		# src,prev = self.pad(self.source,self.prev)
		ob = [self.pad(self.source),self.pad(self.previous)]
		episode_over = self.is_done(action)
		# ob = np.swapaxes(np.array(ob).T,1,-1)
		return ob, reward, episode_over, {}

	def is_done(self,action):     
		if action == eos_token:
			return True
		return False

	def reset(self):
		self.previous = self.target[:-1*self.n_missing_words-1] #Extra -1 because last token is <eos>
		ob = [self.pad(self.source),self.pad(self.previous)]
		return np.swapaxes(np.array(ob).T,1,-1)

	def _render(self, mode='human', close=False):
		pass

	def take_action(self,action):
		self.previous = self.previous.append(action)

			

	def get_reward(self,action):
		if action != eos_token:
			return 0
		else:
			sen_t = []
			sen_g = []

			for i in range(len(self.target)):
				sen_t.append(self.index2word[self.target[i][0]])

			for i in range(len(self.previous)):
				sen_g.append(self.index2word[self.previous[i][0]])
			sen_t = ' '.join(sen_t)
			sen_g = ' '.join(sen_g)
			return sentence_bleu(sen_t,sen_g)


	@property
	def action_space(self):
		return self.action
	@property
	def observation_space(self):
		return self.observation

