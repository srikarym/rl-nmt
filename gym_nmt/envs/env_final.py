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

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

arg  = AttrDict()
arg.update({'task':'translation'})
arg.update({'data':['data/iwslt14.tokenized.de-en']})
arg.update({'lazy_load':False})
arg.update({'left_pad_source':False})
arg.update({'left_pad_target':False})
arg.update({'source_lang':None})
arg.update({'target_lang':None})
arg.update({'raw_text':False})
arg.update({'train_subset':'train'})
arg.update({'valid_subset':'valid'})
arg.update({'max_source_positions':1024})
arg.update({'max_target_positions':1024})
task = fairseq.tasks.setup_task(arg)
task.load_dataset('train')

epoch_itr = task.get_batch_iterator(
	dataset=task.dataset(arg.train_subset),
	max_tokens=4000,
	max_sentences=1,
	max_positions=(100,100),
	ignore_invalid_inputs=True,
	required_batch_size_multiple=1,
	
)
train_data = list(epoch_itr.next_epoch_itr())

class NMTEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	n_vocab = len(task.target_dictionary)
	max_len = 100

	def __init__(self):
		self.task = task
		self.train_data = train_data
		self.previous = None
		self.source = None
		self.target = None
		self.action = spaces.Discrete(self.n_vocab)
		self.observation = np.ones((2,self.max_len))
		
	def init_words(self,n_missing_words):
		self.n_missing_words = n_missing_words

	def seed(self, seed=None): #Don't know how this works
		self.np_random, seed = seeding.np_random(seed)
		return [seed]



	def step(self, action):     #Returns [source, all the previously generated tokens], reward, episode_over, {}

		self.take_action(action)

		reward,counts = self.get_reward(action)
		ob = [self.source,self.previous]
		episode_over = self.is_done(action)
		info = {'True prediction':counts[0],'Total':counts[1]}
		return np.array(ob), reward, episode_over, info

	def is_done(self,action):     
		if action == self.task.target_dictionary.eos():
			return True
		return False

	def reset(self):

		
		training_pair = random.sample(self.train_data,1)[0]

		self.source = training_pair['net_input']['src_tokens'].numpy().tolist()[0]
		self.target = training_pair['target'].numpy().tolist()[0]
		self.generation = []

		if len(self.target)- 1<= self.n_missing_words:
			self.previous = [self.task.target_dictionary.eos()]
		else:
			self.previous = training_pair['net_input']['prev_output_tokens'].numpy().tolist()[0][:-1*self.n_missing_words] #-1 to avoid fullstop 
			self.previous = training_pair['net_input']['prev_output_tokens'].numpy().tolist()[0][:-1*self.n_missing_words] 
		return np.array([self.source,self.previous])

	def _render(self, mode='human', close=False):
		pass

	def take_action(self,action):
		self.previous.append(int(action))
		self.generation.append(int(action))
			

	def get_reward(self,action):
		if action != self.task.target_dictionary.eos():
			return 0,[0,0]
		else:
			
			missing_target = self.target[-1*self.n_missing_words-1:]
			
			tp = 0
			total = len(self.generation)

			for i in range(min(len(missing_target),len(self.generation))):
				if missing_target[i] == self.generation[i]:
					tp+=1


			sen_t = self.task.tgt_dict.string(torch.tensor(missing_target),bpe_symbol='@@ ')
			sen_g = self.task.tgt_dict.string(torch.tensor(self.generation),bpe_symbol='@@ ')


			if (self.n_missing_words == 1):
				total = 2
				if (self.target[-2] == self.generation[0]):
					reward = 100
					tp = 1
				else:
					tp = 0
					reward = 0

			elif (self.n_missing_words == 2):
				total = 4
				if (sorted(self.target[-3:-1]) == sorted(self.generation[:2])):
					reward = 100
					tp = 2
				elif(self.target[-3] == self.generation[0]):
					reward = 50
					tp = 1
				else:
					tp = 0
					reward = 0

			elif(self.n_missing_words == 3):
				total = 6
				if (sorted(self.target[-4:-1]) == sorted(self.generation[:3])):
					reward = 100
					tp = 3
				elif(sorted(self.target[-3:-1]) == sorted(self.generation[:2])):
					reward = 66
					tp = 2
				elif(self.target[-3] == self.generation[0]):
					reward = 33
					tp = 1
				else:
					tp = 0
					reward = 0

			else:
				reward = sentence_bleu(sen_t,sen_g)

			return reward,[tp,total]


	@property
	def action_space(self):
		return self.action
	@property
	def observation_space(self):
		return self.observation

