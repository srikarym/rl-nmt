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
import fairseq
import string
from sys import exit

args = get_args()

random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def checkpoint(epoch,name = None):
	extra_name = args.env_name+"_npro_"+str(args.num_processes)+"_bs_"\
			 +str(args.ppo_batch_size)+"_nsen_"+str(args.num_sentences)+"_seed_"+str(args.seed)
	if not os.path.exists(os.path.join(args.save_dir, extra_name)):
		os.makedirs(os.path.join(args.save_dir, extra_name))

	state = {
		'epoch': epoch,
		'state_dict': actor_critic.state_dict(),
		'optimizer': agent.optimizer.state_dict(),
		
	}
	if name is not None:
		torch.save(state, os.path.join(args.save_dir, extra_name )+ "/model_" + name)
	else:
		torch.save(state, os.path.join(args.save_dir, extra_name )+ "/model_epoch" + str(epoch))


if args.use_wandb:
	wandb.init(project=args.wandb_name)
	config = wandb.config

	config.batch_size = args.ppo_batch_size
	config.num_processes = args.num_processes
	config.lr = args.lr
	config.seed = args.seed
	config.num_steps = args.num_steps
	config.num_sentences = args.num_sentences
	config.seed = args.seed
	wandb.run.description = "{}sen_{}seed_{}lr".format(args.num_sentences,args.seed,args.lr)
	wandb.run.save()
	
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

if (args.reduced):
	src = fairseq.data.Dictionary()
	tgt = fairseq.data.Dictionary()

	data = []

	for i in range(args.num_sentences):
		ssen = train_data[i]['net_input']['src_tokens'][0].numpy().tolist()
		tsen = train_data[i]['target'][0].numpy().tolist()
		
		for s in ssen:
			src.add_symbol(task.src_dict[s])
		for t in tsen:
			tgt.add_symbol(task.tgt_dict[t])

		
	for i in range(args.num_sentences):
		d = {}
		d['id']=i
		ssen = train_data[i]['net_input']['src_tokens'][0].numpy().tolist()
		tsen = train_data[i]['target'][0].numpy().tolist()
		
		new_ssen = []
		for s in ssen:
			word = task.src_dict[s]
			ind = src.index(word)
			new_ssen.append(ind)
		
		new_tsen = []
		for t in tsen:
			word = task.tgt_dict[t]
			ind = tgt.index(word)
			new_tsen.append(ind)
		
		prev = [2] + new_tsen[:-1]
		
		prev = np.array(prev).reshape(1,len(prev))
		new_ssen = np.array(new_ssen).reshape(1,len(new_ssen))
		new_tsen = np.array(new_tsen).reshape(1,len(new_tsen))
		

		d['net_input'] = {'src_tokens':torch.tensor(new_ssen),'prev_output_tokens':torch.tensor(prev)}
		d['target'] = torch.tensor(new_tsen)
		data.append(d)

	train_data = data

	task = AttrDict()

	task.update({'source_dictionary':src})
	task.update({'src_dict':src})
	task.update({'target_dictionary':tgt})
	task.update({'tgt_dict':tgt})


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

envs = [make_env(args.env_name, training_scheme[0],args.seed+i)
		for i in range(args.num_processes)]

envs = SubprocVecEnv(envs)
envs = VecPyTorch(envs, 'cuda',task.source_dictionary.pad())

dummy = gym.make(args.env_name)
dummy.init_words(training_scheme[0],train_data[:args.num_sentences],task)

base_kwargs = {'recurrent': False, 'dummyenv': dummy, 'n_proc': args.num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space, 'Attn', base_kwargs)

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

rollouts = RolloutStorage(args.num_steps, args.num_processes,
						  envs.observation_space.shape, envs.action_space)
rollouts.to(device)
print('Started training')
obs, tac = envs.reset()
idx = tac[:,1]
tac = tac[:,0]
rollouts.obs_s[0].copy_(obs[0])
rollouts.obs_t[0].copy_(obs[1])

eval_reward_best = -100
n_missing_words = training_scheme[0]
eval_episode_rewards = 0
for epoch in range(args.n_epochs + 1):

	value_loss_epoch = 0.0
	action_loss_epoch = 0.0
	dist_entropy_epoch = 0.0
	mean_reward_epoch = 0.0
	total_loss_epoch = 0.0

	ranks_epoch = 0.0

	start = time.time()

	rewards = []
	ranks_iter = []


	if (epoch % args.n_epochs_per_word == 0 and epoch != 0) or eval_episode_rewards>0.96:
		n_missing_words+=1
		print('Num of missing words is',n_missing_words)
		envs.close()
		envs = [make_env(args.env_name, n_missing_words,args.seed+i)
				for i in range(args.num_processes)]
		envs = SubprocVecEnv(envs)
		envs = VecPyTorch(envs, 'cuda', task.source_dictionary.pad())

		rollouts = RolloutStorage(args.num_steps,  args.num_processes,
								  envs.observation_space.shape, envs.action_space)
		rollouts.to(device)

		obs, tac = envs.reset()
		idx = tac[:,1]
		tac = tac[:,0]
		rollouts.obs_s[0].copy_(obs[0])
		rollouts.obs_t[0].copy_(obs[1])

	for step in range(args.num_steps):

		with torch.no_grad():
			value, action, action_log_prob, ranks = actor_critic.act((rollouts.obs_s[step],rollouts.obs_t[step]), tac)
		ranks_iter.append(np.mean(ranks))

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
	value_loss, action_loss, dist_entropy,total_loss = agent.update(rollouts)
	rollouts.after_update()
	end = time.time()

	value_loss_epoch += value_loss
	action_loss_epoch += action_loss
	dist_entropy_epoch += dist_entropy
	total_loss_epoch += total_loss
	mean_reward_epoch += np.mean(rewards)
	ranks_epoch += np.mean(ranks_iter)

	if epoch % args.save_interval == 0 and epoch != 0:
		checkpoint(epoch)

	if (args.eval_interval is not None and epoch%args.eval_interval == 0):
		eval_envs = [make_env(args.env_name, n_missing_words,args.seed+i) for i in range(args.num_processes)]
		eval_envs = SubprocVecEnv(eval_envs)
		eval_envs = VecPyTorch(eval_envs, 'cuda', task.source_dictionary.pad())
		eval_episode_rewards = []

		obs,_ = eval_envs.reset()

		for i in range(2*n_missing_words+1):

			with torch.no_grad():
				_,action,_,_ = actor_critic.act(obs, tac=None,deterministic=True)

			obs_new, reward, done, _ = eval_envs.step(action)
			
			for j in range(args.num_processes):
				print('source sentence is',task.src_dict.string(obs[0][j].long(), bpe_symbol='@@ ').replace('@@ ','').replace('<pad>',''))
				print('target sentence is',task.tgt_dict.string(obs[1][j].long(), bpe_symbol='@@ ').replace('@@ ','').replace('<pad>',''))
				print('True action is',task.tgt_dict[int(tac[j])])
				print('action predicted by the model is',task.tgt_dict[int(action[j].cpu().numpy()[0].tolist())])
				print('rewards are',reward[j])
				print()

			obs = obs_new


			rews = []
			for j in range(len(done)):
				if done[j]:
					rews.append(reward.squeeze(1).cpu().numpy().tolist()[j])

			eval_episode_rewards.extend(rews)

		eval_envs.close()

		print(" Evaluation using {} episodes: mean reward {:.5f}\n".
			format(len(eval_episode_rewards),
				   np.mean(eval_episode_rewards)))
		eval_episode_rewards = np.mean(eval_episode_rewards)

		if eval_episode_rewards > eval_reward_best:
			eval_reward_best = eval_episode_rewards
			checkpoint(epoch,"best")

		if args.use_wandb:
			wandb.log({"Value loss ": value_loss_epoch ,
					   "Action loss": action_loss_epoch ,
					   "Dist entropy": dist_entropy_epoch ,
					   "Mean reward": mean_reward_epoch ,
					   "Mean rank": ranks_epoch,
					   "Total loss": total_loss_epoch ,
					   "Mean evaluation reward": eval_episode_rewards,
					   "Best eval reward":eval_reward_best,
					   "Num of missing words":n_missing_words})




