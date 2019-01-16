import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding
import torch
from sacrebleu import sentence_bleu 
# eos_token = 1
import copy
import glob
import os
import time
from collections import deque
import unicodedata
import string
import re
import random
import torchnlp
from torchnlp.datasets import iwslt_dataset
from torch.autograd import Variable
import fairseq

train_data = iwslt_dataset(train=True)


USE_CUDA = True
class Lang:
    def __init__(self, name):
        self.name = name
        self.dict = fairseq.data.Dictionary()
        self.len_largest = 0
        
    def index_words(self, sentence):
        sentence = sentence.split(' ')
        if len(sentence) > self.len_largest:
            self.len_largest = len(sentence)
            self.largest = sentence
        for word in sentence:
            _ = self.dict.add_symbol(word)

    

def read_langs(lang1, lang2, reverse=False):

	if reverse:
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)
		
	return input_lang, output_lang

def prepare_data(lang1_name, lang2_name,train_data, reverse=False):
    source_lang = Lang(lang1_name)
    target_lang = Lang(lang2_name)

    for pair in train_data:
        source_lang.index_words(pair[lang1_name])
        target_lang.index_words(pair[lang2_name])

    return source_lang, target_lang

def maxlen(pair):
    return max(len(pair[0]),len(pair[1]))

def indexes_from_sentence(lang, sentence):
	return [lang.dict.index(word) for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
	indexes = indexes_from_sentence(lang, sentence)
	indexes.append(lang.dict.eos())
	return indexes

def variables_from_pair(input_lang, output_lang,pair,lang1,lang2):
	input_variable = variable_from_sentence(input_lang, pair[lang1])
	target_variable = variable_from_sentence(output_lang, pair[lang2])
	return (input_variable, target_variable)



class NMTEnvRed(gym.Env):
	metadata = {'render.modes': ['human']}
	source_lang,target_lang = prepare_data('en','de',train_data)
	n_vocab = len(target_lang.dict.symbols)
	max_len = target_lang.len_largest

	def __init__(self):
		self.data = train_data
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
		if action == self.source_lang.dict.eos():
			return True
		return False

	def reset(self):

		
		training_pair = variables_from_pair(self.source_lang, self.target_lang,random.choice(train_data),'en','de')

		while maxlen(training_pair)>100:
			training_pair = variables_from_pair(self.source_lang, self.target_lang,random.choice(train_data),'en','de')

		self.source = training_pair[0]
		self.target = training_pair[1]
		self.generation = []

		if len(self.target)- 1<= self.n_missing_words:
			self.previous = [self.source_lang.dict.eos()]
		else:
			self.previous = self.target[:-1*self.n_missing_words-1] #Extra -1 because last token is <eos>
		return np.array([self.source,self.previous])

	def _render(self, mode='human', close=False):
		pass

	def take_action(self,action):
		self.previous.append(int(action))
		self.generation.append(int(action))
			

	def get_reward(self,action):
		if action != self.source_lang.dict.eos():
			return 0,[0,0]
		else:
			
			missing_target = self.target[-1*self.n_missing_words-1:]
			
			tp = 0
			# fp = 0
			total = len(self.generation)

			for i in range(len(self.generation)):
				if (i<len(missing_target)):
					if missing_target[i] == self.generation[i]:
						tp+=1
				else:
					if self.generation[i] == self.source_lang.dict.eos():
						tp+=1


			sen_t = []
			sen_g = []

			# sen_t = self.target_lang.dict.string(torch.tensor(self.target))
			# sen_g = self.target_lang.dict.string(torch.tensor(self.previous))

			sen_t = self.target_lang.dict.string(torch.tensor(missing_target))
			sen_g = self.target_lang.dict.string(torch.tensor(self.generation))


			if (self.n_missing_words == 1):
				if (self.target[:-2] in self.generation):
					reward = 100
				else:
					reward = 0

			elif (self.n_missing_words == 2):
				common = len(list(set(target[-3:-1]) & self.generation))
				reward = 50*common
			elif(self.n_missing_words == 3):
				common = len(list(set(target[-4:-1]) & self.generation))
				reward = 33.3*common
			else:
				reward = sentence_bleu(sen_t,sen_g)

			return reward,[tp,total]


	@property
	def action_space(self):
		return self.action
	@property
	def observation_space(self):
		return self.observation

