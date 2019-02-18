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

def random_string(N):
	return ''.join(random.choice(string.ascii_lowercase) for _ in range(N))


def create_data(num_sens=10,len_vocab=100,min_len = 10,max_len = 20):
	
	data = []
	for i in range(num_sens):
		d = {}
		d['id']=i
		src = np.random.randint(4,len_vocab,(1,random.choice(range(min_len,max_len))))
		src = np.insert(src,0,2,1)
		tg = np.random.randint(4,len_vocab,(1,random.choice(range(min_len,max_len))))
		prev = np.insert(tg,0,2,1)
		target = np.append(tg,2)
		target = target.reshape(1,len(target))
		
		d['net_input'] = {'src_tokens':torch.tensor(src),'prev_output_tokens':torch.tensor(prev)}
		d['target'] = torch.tensor(target)
		data.append(d)
	return data

print('Loading data')

len_vocab = 100
num_sentences = args.num_sentences

src = fairseq.data.Dictionary()
tgt = fairseq.data.Dictionary()

task = AttrDict()

task.update({'source_dictionary':src})
task.update({'src_dict':src})
task.update({'target_dictionary':tgt})
task.update({'tgt_dict':tgt})


for _ in range(len_vocab-4):
	src.add_symbol(random_string(random.choice(range(3,8))))
	tgt.add_symbol(random_string(random.choice(range(3,8))))


train_data = create_data(num_sentences,len_vocab)

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

	# log = deepcopy(log_dict)

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

		# for j in range(args.num_processes):
		#     log[idx[j]]['action'].append(task.tgt_dict[int(action[j].cpu().numpy()[0].tolist())])

		#     log[idx[j]]['tac'] = task.tgt_dict[int(tac[j])]

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
	# if (args.num_sentences!= -1):
	#     for j in range(args.num_sentences):
	#         index = indexes[j]
	#         print('\nSentence pair {}, source sentence -  {},\n target sentence - {}'\
	#             .format(j,log[index]['source'],log[index]['target']))
	#         print('True action is',log[index]['tac'])
	#         print('Percentage of true action is',log[index]['action'].count(log[index]['tac'])*100/len(log[index]['action']))
	#         print('Actions predicted by the model are',log[index]['action'])

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
					 +str(args.ppo_batch_size)+"_nsen_"+str(args.num_sentences)+"_seed_"+str(args.seed)
		if not os.path.exists(os.path.join(args.save_dir, extra_name)):
			os.makedirs(os.path.join(args.save_dir, extra_name))

		state = {
			'epoch': epoch,
			'state_dict': actor_critic.state_dict(),
			'optimizer': agent.optimizer.state_dict(),
			
		}
		torch.save(state, os.path.join(args.save_dir, extra_name )+ "/model_epoch" + str(epoch))

	if (args.eval_interval is not None and epoch%args.eval_interval == 0):
		eval_envs = [make_env(args.env_name, training_scheme[0],args.seed+i) for i in range(args.num_processes)]
		eval_envs = SubprocVecEnv(eval_envs)
		eval_envs = VecPyTorch(eval_envs, 'cuda', task.source_dictionary.pad())
		eval_episode_rewards = []

		obs,tac = eval_envs.reset()
		idx = tac[:,1]
		tac = tac[:,0]


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

		masks = torch.FloatTensor([[0.0] if done_ else [1.0]
								   for done_ in done])

		eval_episode_rewards.extend(reward.squeeze(1).cpu().numpy().tolist())

		eval_envs.close()

		print(" Evaluation using {} episodes: mean reward {:.5f}\n".
			format(len(eval_episode_rewards),
				   np.mean(eval_episode_rewards)))

		if args.use_wandb:
			wandb.log({"Mean evaluation reward": np.mean(eval_episode_rewards) })
