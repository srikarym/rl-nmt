import os
import numpy as np
import torch
import time
import sys
import wandb
sys.path.insert(0, 'pytorch-a2c-ppo-acktr')
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args
import torch.nn as nn
os.environ['OMP_NUM_THREADS'] = '1'
import string
from sys import exit
from dataloader import load_train_data,AttrDict
from env_utils import make_vec_envs, make_dummy
from myutils import logger
import random
from fairseq.utils import _upgrade_state_dict

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

	wandb.config.update(args)

	wandb.run.description = args.run_name
	wandb.run.save()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data,task = load_train_data()

envs = make_vec_envs(args.env_name,args.n_words,args.seed,train_data,task,args.num_processes)

dummy = make_dummy(args.env_name,args.n_words,train_data,task)

base_kwargs = {'recurrent': False, 'dummyenv': dummy, 'n_proc': args.num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space, 'Attn', base_kwargs)

actor_critic.to(device)

agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.ppo_batch_size, \
				args.ppo_mini_batches, args.value_loss_coef, args.entropy_coef, lr=args.lr,eps=args.eps,\
				max_grad_norm=args.max_grad_norm)

if (args.checkpoint): #Load from checkpoint
	state = torch.load(args.file_path)
	state = _upgrade_state_dict(state)
	actor_critic.base.model.upgrade_state_dict(state['model'])
	actor_critic.base.model.load_state_dict(state['model'], strict=True)



n_epochs_currentword = 0

ratio = 0.5
rollouts = RolloutStorage(args.num_steps, args.num_processes,
						  envs.observation_space.shape, envs.action_space,ratio,1)
rollouts.to(device)
print('Started training')
obs, info = envs.reset()
# idx = tac[:,1]
tac = info[:,0]
rollouts.obs_s[0].copy_(obs[0])
rollouts.obs_t[0].copy_(obs[1])

n_words = args.n_words

epochs_per_word = args.n_epochs_per_word

for epoch in range(args.n_epochs):
	n_epochs_currentword += 1
	value_loss_epoch = 0.0
	action_loss_epoch = 0.0
	dist_entropy_epoch = 0.0
	nll_loss_epoch = 0.0
	mean_reward_epoch = 0.0
	total_loss_epoch = 0.0

	ranks_epoch = 0.0

	start = time.time()

	rewards = []
	ranks_iter = []

	use_nll = False

	# if n_epochs_currentword > args.n_epochs_per_word//2 :
	# 	use_nll = False
	# else:
	# 	use_nll = True

	if  n_epochs_currentword > args.n_epochs_per_word :
		n_epochs_currentword = 0
		n_words += args.nwwords_back 
		print('Num of missing words is',n_words)
		envs.transition(args.nwwords_back)
		# envs = make_vec_envs(args.env_name,n_words,args.seed,train_data[:args.num_sentences],task,args.num_processes)

		# rollouts = RolloutStorage(args.num_steps,  args.num_processes,
		# 						  envs.observation_space.shape, envs.action_space,ratio,n_words)
		# rollouts.to(device)

		obs, info = envs.reset()
		tac = info[:,0]
		rollouts.obs_s[0].copy_(obs[0])
		rollouts.obs_t[0].copy_(obs[1])

	for step in range(args.num_steps):

		with torch.no_grad():
			value, action, action_log_prob, ranks = actor_critic.act((rollouts.obs_s[step],rollouts.obs_t[step]), tac)
		# print(ranks)
		ranks_iter.append(np.mean(ranks))

		obs, reward, done, info = envs.step(action,use_nll)

		tac = info[:,0]

		masks = torch.FloatTensor([[0.0] if done_ else [1.0]
								   for done_ in done])

		rollouts.insert(obs,action, action_log_prob, value, reward, masks,tac)
		rewards.append(np.mean(reward.squeeze(1).cpu().numpy()))

	with torch.no_grad():
		next_value = actor_critic.get_value((rollouts.obs_s[-1],rollouts.obs_t[-1])).detach()
		# next_value = 0

	rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
	value_loss, action_loss, dist_entropy, nll_loss, total_loss = agent.update(rollouts,use_nll)
	rollouts.after_update()
	end = time.time()

	speed = (args.num_steps*args.num_processes)/(end-start)

	value_loss_epoch += value_loss
	action_loss_epoch += action_loss
	dist_entropy_epoch += dist_entropy
	nll_loss_epoch += nll_loss
	total_loss_epoch += total_loss
	mean_reward_epoch += np.mean(rewards)
	ranks_epoch += np.mean(ranks_iter)

	if epoch % args.save_interval == 0 and epoch != 0:
		checkpoint(epoch)

	#Calculate bleu score
	with torch.no_grad():
		corpus_bleu, sentence_bleu = actor_critic.bleuscore()


	if (args.use_wandb):
		logger.log(epoch,n_words,value_loss_epoch,action_loss_epoch,dist_entropy_epoch, nll_loss_epoch,mean_reward_epoch,\
			ranks_epoch,total_loss_epoch,speed,corpus_bleu, sentence_bleu)
