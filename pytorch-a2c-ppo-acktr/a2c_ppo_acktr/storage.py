import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _flatten_helper(_tensor):
	T,N = _tensor.size()[:2]
	return _tensor.view(T * N, -1)


class RolloutStorage(object):
	def __init__(self, num_steps, num_processes, obs_shape, action_space,ratio,n_words):
		# self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
		self.obs_s = torch.ones(num_steps+1,num_processes,100)
		self.obs_t = torch.ones(num_steps+1,num_processes,100)

		self.gt = torch.ones(num_steps+1,num_processes)


		self.num_processes = num_processes
		self.num_steps = num_steps
		# self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
		self.rewards = torch.zeros(num_steps , num_processes, 1)
		self.value_preds = torch.zeros(num_steps  +1, num_processes, 1)
		self.returns = torch.zeros(num_steps+1,num_processes,1)
		self.action_log_probs = torch.zeros(num_steps , num_processes, 1)
		if action_space.__class__.__name__ == 'Discrete':
			action_shape = 1
		else:
			action_shape = action_space.shape[0]
		self.actions = torch.zeros(num_steps , num_processes, action_shape)
		if action_space.__class__.__name__ == 'Discrete':
			self.actions = self.actions.long()
		self.masks = torch.ones(num_steps + 1, num_processes, 1)

		self.ratio = ratio
		self.n_words = n_words

		self.step = 0

	def to(self, device):
		# self.obs = self.obs.to(device)
		# self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
		self.obs_s.to(device)
		self.obs_t.to(device)
		self.rewards = self.rewards.to(device)
		self.value_preds = self.value_preds.to(device)
		self.returns = self.returns.to(device)
		self.action_log_probs = self.action_log_probs.to(device)
		self.actions = self.actions.to(device)
		self.masks = self.masks.to(device)
		self.returns = self.returns.to(device)


	def insert(self,obs, actions, action_log_probs, value_preds, rewards, masks,gt):

		self.obs_s[self.step + 1].copy_(obs[0])
		self.obs_t[self.step+1].copy_(obs[1])
		self.actions[self.step].copy_(actions)
		self.action_log_probs[self.step].copy_(action_log_probs)
		self.value_preds[self.step].copy_(value_preds)
		self.rewards[self.step].copy_(rewards)
		self.masks[self.step+1].copy_(masks)

		# print(gt.shape)

		self.gt[self.step + 1].copy_(torch.tensor(gt))

		self.step = (self.step + 1) % (self.num_steps)


	def after_update(self):
		self.obs_s[0].copy_(self.obs_s[-1])
		self.obs_t[0].copy_(self.obs_t[-1])
		self.masks[0].copy_(self.masks[-1])
		self.gt[0].copy_(self.gt[-1])

	def compute_returns(self, next_value, use_gae, gamma, tau):
		if use_gae:
			self.value_preds[-1] = next_value
			gae = 0
			for step in reversed(range(self.rewards.size(0))):
				delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
						self.value_preds[step]
				gae = delta + gamma * tau * self.masks[step + 1] * gae
				self.returns[step] = gae + self.value_preds[step]
		else:
			self.returns[-1] = next_value
			for step in reversed(range(self.rewards.size(0))):
				self.returns[step] = self.returns[step + 1] * \
									 gamma * self.masks[step + 1] + self.rewards[step]

	def feed_forward_generator(self, advantages, mini_batch_size):


		obs_s_flat = _flatten_helper(self.obs_s[:-1])
		obs_t_flat = _flatten_helper(self.obs_t[:-1])


		batch_size = obs_s_flat.shape[0]
		sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),mini_batch_size,drop_last = False)

		gt_flat = self.gt[:-1].view(-1,1)

		actions_flat = self.actions.view(-1, 1)
		value_preds_flat = self.value_preds[:-1].view(-1, 1)
		returns_flat = self.returns[:-1].view(-1, 1)
		action_log_probs_flat = self.action_log_probs.view(-1, 1)
		advantages_flat = advantages.view(-1, 1)

		count = 0

		for indices in sampler:

			# print('total number of obs is',obs_s_flat.shape[0])
			# print('indices are',indices)
			count += 1
			if count > 5:
				break

			obs_batch_s = obs_s_flat[indices]
			obs_batch_t = obs_t_flat[indices]


			actions_batch = actions_flat[indices]
			value_preds_batch = value_preds_flat[indices]
			return_batch = returns_flat[indices]
			old_action_log_probs_batch = action_log_probs_flat[indices]
			adv_targ = advantages_flat[indices]
			gt_batch = gt_flat[indices]

			yield (obs_batch_s, obs_batch_t), actions_batch, \
				   value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ,gt_batch


