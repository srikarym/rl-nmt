import numpy as np
import gym
import gym.spaces as spaces
from gym.utils import seeding
import torch
from sacrebleu import sentence_bleu 
eos_token = 1
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

train_data = iwslt_dataset(train=True)

SOS_token = 0
EOS_token = 1
USE_CUDA = True
class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2 # Count SOS and EOS
		self.len_largest = 0
		self.largest = ""
	  
	def index_words(self, sentence):
		sentence = sentence.split(' ')
		if len(sentence) > self.len_largest:
			self.len_largest = len(sentence)
			self.largest = sentence
		for word in sentence:
			self.index_word(word)

	def index_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1
def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
	s = unicode_to_ascii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def read_langs(lang1, lang2, reverse=False):

	if reverse:
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)
		
	return input_lang, output_lang

def prepare_data(lang1_name, lang2_name,train_data, reverse=False):
	input_lang, output_lang = read_langs(lang1_name, lang2_name, reverse)
	
	
	for pair in train_data:
		input_lang.index_words(pair[lang1_name])
		output_lang.index_words(pair[lang2_name])

	return input_lang, output_lang


def indexes_from_sentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
	indexes = indexes_from_sentence(lang, sentence)
	indexes.append(EOS_token)
	var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
	if USE_CUDA: var = var.cuda()
	return var

def variables_from_pair(input_lang, output_lang,pair,lang1,lang2):
	input_variable = variable_from_sentence(input_lang, pair[lang1])
	target_variable = variable_from_sentence(output_lang, pair[lang2])
	return (input_variable, target_variable)



class NMTEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	input_lang, output_lang = prepare_data('en', 'de',train_data, True)
	n_vocab = output_lang.n_words
	max_len = output_lang.len_largest

	def __init__(self):
		self.previous = None
		self.source = None
		self.target = None
		self.action = spaces.Discrete(self.n_vocab)
		self.observation = np.ones((2,self.max_len))


	def pad(self,vec):
		vec = np.pad(vec[:,0],(0,self.max_len-len(vec)), 'constant',constant_values=-1).reshape(self.max_len,1 )
		return vec



	def seed(self, seed=None): #Don't know how this works
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def step(self, action):     #Returns [source, all the previously generated tokens], reward, episode_over, {}

		self.take_action(action)

		reward = self.get_reward(action)
		ob = [self.pad(self.source),self.pad(self.previous)]
		episode_over = self.is_done(action)
		ob = np.swapaxes(np.array(ob).T,1,-1)
		return torch.tensor(ob).cuda(), reward, episode_over, {}

	def is_done(self,action):     
		if action == EOS_token:
			return True
		return False

	def reset(self,n_missing_words):
		self.n_missing_words = n_missing_words
		training_pair = variables_from_pair(self.input_lang, self.output_lang,random.choice(train_data),'en','de')

		self.source = training_pair[0].cpu().numpy()
		self.target = training_pair[1].cpu().numpy()

		self.previous = self.target[:-1*self.n_missing_words-1] #Extra -1 because last token is <eos>
		
		ob = [self.pad(self.source),self.pad(self.previous)]
		ob =  np.swapaxes(np.array(ob).T,1,-1)
		return torch.tensor(ob).cuda()
	def _render(self, mode='human', close=False):
		pass

	def take_action(self,action):
		action = action.cpu().numpy()[0][0]
		self.previous = np.append(self.previous,action).reshape(len(self.previous)+1,1)

			

	def get_reward(self,action):
		if action != EOS_token:
			return 0
		else:
			sen_t = []
			sen_g = []

			for i in range(len(self.target)):
				sen_t.append(self.output_lang.index2word[self.target[i][0]])

			for i in range(len(self.previous)):
				sen_g.append(self.output_lang.index2word[self.previous[i][0]])
			sen_t = ' '.join(sen_t)
			sen_g = ' '.join(sen_g)
			return sentence_bleu(sen_t,sen_g)


	@property
	def action_space(self):
		return self.action
	@property
	def observation_space(self):
		return self.observation

