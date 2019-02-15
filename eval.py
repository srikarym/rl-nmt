import os
import gym_nmt
import gym
import numpy as np
import torch
import time
import sys
import wandb
from modified_subproc import SubprocVecEnv
sys.path.insert(0, 'pytorch-a2c-ppo-acktr')
from utils import VecPyTorch
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args
import gc
import _pickle as pickle
import torch.nn as nn
from copy import deepcopy
os.environ['OMP_NUM_THREADS'] = '1'
import random

args = get_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


if args.use_wandb:
	wandb.init(project=args.wandb_name)
	config = wandb.config

	config.batch_size = args.ppo_batch_size
	config.num_processes = args.num_processes
	config.lr = args.lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def load_cpickle_gc(fname):
	output = open(fname, 'rb')

	gc.disable()
	mydict = pickle.load(output)
	gc.enable()
	output.close()
	return mydict

print('Loading data')

train_data = load_cpickle_gc('data/data.pkl')
task = load_cpickle_gc('data/task.pkl')


print('Data loaded')


def make_env(env_id, n_missing_words,seed):
	def _thunk():
		env = gym.make(env_id)
		env.init_words(n_missing_words,train_data[:args.num_sentences],task)
		env.seed()

		return env

	return _thunk

training_scheme = []

for i in range(1, args.max_missing_words):
	training_scheme.extend([i] * args.n_epochs_per_word)


dummy = gym.make(args.env_name)
dummy.init_words(training_scheme[0],train_data[:args.num_sentences],task)

base_kwargs = {'recurrent': False, 'dummyenv': dummy, 'n_proc': args.num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space, 'Attn', base_kwargs)

actor_critic.to(device)

agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.ppo_batch_size,
				 args.value_loss_coef, args.entropy_coef, lr=args.lr,
				 eps=args.eps,
				 max_grad_norm=args.max_grad_norm)

state = torch.load(args.file_path)
actor_critic.load_state_dict(state['state_dict'])
agent.optimizer.load_state_dict(state['optimizer'])

eval_envs = [make_env(env_id=args.env_name, n_missing_words=n_missing_words) for i in range(args.num_processes)]
eval_envs = SubprocVecEnv(eval_envs)
eval_envs = VecPyTorch(eval_envs, 'cuda', task.source_dictionary.pad())
eval_episode_rewards = []

obs, tac = envs.reset()
idx = tac[:,1]
tac = tac[:,0]

keys=[]
indexes = []
for i in range(args.num_sentences):
	keys.append((train_data[i]['id'].numpy().tolist()[0],train_data[i]['net_input']['src_tokens'],train_data[i]['target']))
	indexes.append(train_data[i]['id'].numpy().tolist()[0])
log_dict = {index: {'source':task.src_dict.string(source, bpe_symbol='@@ ').replace('@@ ',''),\
	'target':task.tgt_dict.string(target, bpe_symbol='@@ ').replace('@@ ',''),'action':[]} for index,source,target in keys}

for j in range(100):
	log = deepcopy(log_dict)
	with torch.no_grad():
		_,action,_,_ = actor_critic.act(obs, tac,deterministic=True)

	for j in range(args.num_processes):
		log[idx[j]]['action'].append(task.tgt_dict[int(action[j].cpu().numpy()[0].tolist())])
		log[idx[j]]['tac'] = task.tgt_dict[int(tac[j])]


	obs, reward, done, tac = envs.step(action)

	idx = tac[:,1]
	tac = tac[:,0]
	masks = torch.FloatTensor([[0.0] if done_ else [1.0]
							   for done_ in done])

	eval_episode_rewards.extend(reward.squeeze(1).cpu().numpy().tolist())

	for j in range(args.num_sentences):
		index = indexes[j]
		print('\nSentence pair {}, source sentence -  {},\n target sentence - {}'\
			.format(j,log[index]['source'],log[index]['target']))
		print('True action is',log[index]['tac'])
		print('Percentage of true action is',log[index]['action'].count(log[index]['tac'])*100/len(log[index]['action']))
		print('Actions predicted by the model are',log[index]['action'])


print(" Evaluation using {} episodes: mean reward {:.5f}\n".
	format(len(eval_episode_rewards),
		   np.mean(eval_episode_rewards)))

