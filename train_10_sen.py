import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gym
import gym_nmt
from modified_subproc import SubprocVecEnv
from utils import VecPyTorch
import numpy as np
import torch
import time
import sys
sys.path.insert(0,'pytorch-a2c-ppo-acktr')

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args
from utils import reshape_batch
from tensorboardX import SummaryWriter

import wandb
args = get_args()
if(args.use_wandb):
	wandb.init(project=args.wandb_name)
	config = wandb.config

	config.batch_size = args.ppo_batch_size
	config.num_processes = args.num_processes
	config.lr = args.lr


writer = SummaryWriter(args.log_dir)

def make_env(env_id, n_missing_words):
	def _thunk():
		env = gym.make(env_id)
		env.init_words(n_missing_words)

		return env

	return _thunk

training_scheme = []


for i in range(1,args.max_missing_words):
	training_scheme.extend([i]*args.n_epochs_per_word)


envs = [make_env(env_id = args.env_name,n_missing_words=training_scheme[0])
			for i in range(args.num_processes)]

envs = SubprocVecEnv(envs)
envs = VecPyTorch(envs,'cuda')


base_kwargs={'recurrent': False,'dummyenv':envs.dummyenv,'n_proc':args.num_processes}
actor_critic = Policy(envs.observation_space.shape, envs.action_space,'Attn',base_kwargs)
if(args.use_wandb):
	wandb.watch(actor_critic)
# wandb.watch(actor_critic)


agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.ppo_batch_size,
				 args.value_loss_coef, args.entropy_coef, lr=args.lr,
					   eps=args.eps,
					   max_grad_norm=args.max_grad_norm)

dummy = gym.make(args.env_name)
len_train_data = len(dummy.train_data)

if (args.sen_per_epoch == 0):
	sen_per_epoch = len_train_data//(args.num_steps*args.num_processes)
else:
	sen_per_epoch = args.sen_per_epoch


rollouts = RolloutStorage(args.num_steps,1, args.num_processes,
						envs.observation_space.shape, envs.action_space)
print('Started training')
for epoch in range(args.n_epochs+1):

	value_loss_epoch = 0.0
	action_loss_epoch = 0.0
	dist_entropy_epoch = 0.0
	mean_reward_epoch = 0.0

	truepred_iter = 0
	totalpred_iter = 0
	ranks_epoch = 0.0
	
	for ite in range(sen_per_epoch):	
		start = time.time()

		n_missing_words = training_scheme[epoch]
		
		rewards = []
		ranks_iter = []
		
		if (epoch%args.n_epochs_per_word == 0 and epoch!=0):

			envs = [make_env(env_id = args.env_name,n_missing_words=n_missing_words)
				for i in range(args.num_processes)]
			envs = SubprocVecEnv(envs)
			envs = VecPyTorch(envs,'cuda')

			rollouts = RolloutStorage(args.num_steps,1, args.num_processes,
							envs.observation_space.shape, envs.action_space)


		obs = []
		


		for step in range(args.num_steps):

			ob,tac = envs.reset()
			obs.append(ob)

			with torch.no_grad():
				value, action, action_log_prob,ranks = actor_critic.act(ob,tac)

			ob, reward, done, infos,tac = envs.step(action)
			# obs.append(ob)

			ranks_iter.append(np.mean(ranks))
			masks = torch.FloatTensor([[0.0] if done_ else [1.0]
								   for done_ in done])

			rollouts.insert( action, action_log_prob, value, reward,masks)
			rewards.append(np.mean(reward.squeeze(1).cpu().numpy()))

		rollouts.insert_obs(reshape_batch(obs))
		with torch.no_grad():
			next_value = actor_critic.get_value(ob).detach()

		rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
		end = time.time()
		total_steps = args.num_steps*args.num_processes
		value_loss, action_loss,dist_entropy = agent.update(rollouts)
		writer.add_scalar('Running Value loss',value_loss,ite+epoch*sen_per_epoch)
		writer.add_scalar('Running action loss',action_loss,ite+epoch*sen_per_epoch)
		writer.add_scalar('Running Dist entropy',dist_entropy,ite+epoch*sen_per_epoch)
		writer.add_scalar('Running mean reward ',np.mean(rewards),ite+epoch*sen_per_epoch)
		writer.add_scalar('Steps per sec',total_steps/(end-start),ite+epoch*sen_per_epoch)
		writer.add_scalar('Running rank of predicted actions',np.mean(ranks_iter),ite+epoch*sen_per_epoch)



		value_loss_epoch+=value_loss
		action_loss_epoch+=action_loss
		dist_entropy_epoch+=dist_entropy
		mean_reward_epoch+= np.mean(rewards)
		ranks_epoch+= np.mean(ranks_iter)



	total_loss = args.value_loss_coef*value_loss_epoch + action_loss_epoch - dist_entropy_epoch*args.entropy_coef
		
	writer.add_scalar('Epoch Value loss',value_loss_epoch/sen_per_epoch,epoch)
	writer.add_scalar('Epoch Action loss',action_loss_epoch/sen_per_epoch,epoch)
	writer.add_scalar('Epoch distribution entropy',dist_entropy_epoch/sen_per_epoch,epoch)
	writer.add_scalar('Epoch mean reward',mean_reward_epoch/sen_per_epoch,epoch)
	writer.add_scalar('Epoch mean rank',ranks_epoch/sen_per_epoch,epoch)



	writer.add_scalar('Learning rate',args.lr,epoch)
	writer.add_scalar('Total loss',total_loss/sen_per_epoch,epoch)


	if(args.use_wandb):

		wandb.log({"Value loss ": value_loss_epoch/sen_per_epoch,
				   "Action loss": action_loss_epoch/sen_per_epoch,
				   "Dist entropy": dist_entropy_epoch/sen_per_epoch,
				   "Mean reward":mean_reward_epoch/sen_per_epoch,
				   "Mean rank":ranks_epoch/sen_per_epoch,
				   "Total loss":total_loss/sen_per_epoch})


	if (epoch%args.save_interval == 0 and epoch!=0):
		if not os.path.exists(os.path.join(args.save_dir, args.env_name)):
			os.makedirs(os.path.join(args.save_dir, args.env_name))


		save_model = actor_critic
		torch.save(save_model.state_dict(), os.path.join(args.save_dir, args.env_name +"_epoch"+str(epoch)))



	
