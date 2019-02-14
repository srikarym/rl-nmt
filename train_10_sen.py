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

envs = [make_env(env_id=args.env_name, n_missing_words=training_scheme[0],args.seed+i)
		for i in range(args.num_processes)]

envs = SubprocVecEnv(envs)
envs = VecPyTorch(envs, 'cuda',task.source_dictionary.pad())

dummy = gym.make(args.env_name)
dummy.init_words(training_scheme[0],train_data[:args.num_sentences],task)

base_kwargs = {'recurrent': False, 'dummyenv': dummy, 'n_proc': args.num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space, 'Attn', base_kwargs)
# if args.use_wandb:
#     wandb.watch(actor_critic)
# actor_critic = nn.DataParallel(actor_critic)
actor_critic.to(device)

agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.ppo_batch_size,
				 args.value_loss_coef, args.entropy_coef, lr=args.lr,
				 eps=args.eps,
				 max_grad_norm=args.max_grad_norm)

if (args.checkpoint):
	state = torch.load(args.file_path)
	actor_critic.load_state_dict(state['state_dict'])
	agent.optimizer.load_state_dict(state['optimizer'])

if (args.sen_per_epoch == 0):
	sen_per_epoch = len_train_data // (args.num_steps * args.num_processes)
else:
	sen_per_epoch = args.sen_per_epoch

rollouts = RolloutStorage(args.num_steps, 1, args.num_processes,
						  envs.observation_space.shape, envs.action_space)
rollouts.to(device)
print('Started training')
obs, tac = envs.reset()
idx = tac[:,1]
tac = tac[:,0]
rollouts.obs_s[0].copy_(obs[0])
rollouts.obs_t[0].copy_(obs[1])

keys=[]
indexes = []
for i in range(args.num_sentences):
	keys.append((train_data[i]['id'].numpy().tolist()[0],train_data[i]['net_input']['src_tokens'],train_data[i]['target']))
	indexes.append(train_data[i]['id'].numpy().tolist()[0])
log_dict = {index: {'source':task.src_dict.string(source, bpe_symbol='@@ ').replace('@@ ',''),\
	'target':task.tgt_dict.string(target, bpe_symbol='@@ ').replace('@@ ',''),'action':[]} for index,source,target in keys}

for epoch in range(args.n_epochs + 1):

	value_loss_epoch = 0.0
	action_loss_epoch = 0.0
	dist_entropy_epoch = 0.0
	mean_reward_epoch = 0.0

	ranks_epoch = 0.0

	start = time.time()

	n_missing_words = training_scheme[epoch]

	rewards = []
	ranks_iter = []

	log = deepcopy(log_dict)
	assert log[indexes]['action'] == []

	if epoch % args.n_epochs_per_word == 0 and epoch != 0:
		envs = [make_env(env_id=args.env_name, n_missing_words=n_missing_words)
				for i in range(args.num_processes)]
		envs = SubprocVecEnv(envs)
		envs = VecPyTorch(envs, 'cuda', task.source_dictionary.pad())

		rollouts = RolloutStorage(args.num_steps, 1, args.num_processes,
								  envs.observation_space.shape, envs.action_space)

	for step in range(args.num_steps):

		with torch.no_grad():
			value, action, action_log_prob, ranks = actor_critic.act((rollouts.obs_s[step],rollouts.obs_t[step]), tac)
		ranks_iter.append(np.mean(ranks))

		for j in range(args.num_processes):
			log[idx[j]]['action'].append(task.tgt_dict[int(action[j].cpu().numpy()[0].tolist())])

			log[idx[j]]['tac'] = task.tgt_dict[int(tac[j])]

		obs, reward, done, tac = envs.step(action)


		idx = tac[:,1]
		tac = tac[:,0]

		masks = torch.FloatTensor([[0.0] if done_ else [1.0]
								   for done_ in done])

		rollouts.insert(obs,action, action_log_prob, value, reward, masks)
		rewards.append(np.mean(reward.squeeze(1).cpu().numpy()))

	with torch.no_grad():
		next_value = actor_critic.get_value((rollouts.obs_s[-1],rollouts.obs_t[-1])).detach()

	rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
	value_loss, action_loss, dist_entropy = agent.update(rollouts)
	rollouts.after_update()
	end = time.time()
	total_steps = args.num_steps * args.num_processes * (1)

	value_loss_epoch += value_loss
	action_loss_epoch += action_loss
	dist_entropy_epoch += dist_entropy
	mean_reward_epoch += np.mean(rewards)
	ranks_epoch += np.mean(ranks_iter)

	print('\n\nEpoch {} out of {}, Mean reward is {}'.format(epoch,args.n_epochs,mean_reward_epoch))
	if (args.num_sentences!= -1):
		for j in range(args.num_sentences):
			index = indexes[j]
			print('\nSentence pair {}, source sentence -  {},\n target sentence - {}'\
				.format(j,log[index]['source'],log[index]['target']))
			print('True action is',log[index]['tac'])
			print('Percentage of true action is',log[index]['action'].count(log[index]['tac'])*100/len(log[index]['action']))
			print('Actions predicted by the model are',log[index]['action'])

	total_loss = args.value_loss_coef * value_loss_epoch  + action_loss_epoch - dist_entropy_epoch  * args.entropy_coef


	if args.use_wandb:
		wandb.log({"Value loss ": value_loss_epoch ,
				   "Action loss": action_loss_epoch ,
				   "Dist entropy": dist_entropy_epoch ,
				   "Mean reward": mean_reward_epoch ,
				   "Mean rank": ranks_epoch,
				   "Total loss": total_loss })

	if epoch % args.save_interval == 0 and epoch != 0:
		extra_name = args.env_name+"_npro_"+str(args.num_processes)+"_bs_"\
					 +str(args.ppo_batch_size)+"_nsen_"+str(args.num_sentences)
		if not os.path.exists(os.path.join(args.save_dir, extra_name)):
			os.makedirs(os.path.join(args.save_dir, extra_name))

		state = {
			'epoch': epoch,
			'state_dict': actor_critic.state_dict(),
			'optimizer': agent.optimizer.state_dict(),
			
		}
		torch.save(state, os.path.join(args.save_dir, extra_name )+ "/model_epoch" + str(epoch))
