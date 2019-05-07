import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPO():
	def __init__(self,
				 actor_critic,
				 clip_param,
				 ppo_epoch,
				 mini_batch_size,
				 ppo_mini_batches,
				 value_loss_coef,
				 entropy_coef,
				 lr=None,
				 eps=None,
				 max_grad_norm=None,
				 use_clipped_value_loss=True):

		self.actor_critic = actor_critic

		self.clip_param = clip_param
		self.ppo_epoch = ppo_epoch
		self.mini_batch_size = mini_batch_size

		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef

		self.max_grad_norm = max_grad_norm
		self.use_clipped_value_loss = use_clipped_value_loss

		self.ppo_mini_batches = ppo_mini_batches

		self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

	def update(self, rollouts,usenll = False):

		self.actor_critic.base.train()

		advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

		value_loss_epoch = 0
		action_loss_epoch = 0
		dist_entropy_epoch = 0
		nll_loss_epoch = 0
		total_loss_epoch = 0

		num_updates = 0
		criterion = nn.NLLLoss()
		for e in range(self.ppo_epoch):
			data_generator = rollouts.feed_forward_generator(advantages, self.mini_batch_size, self.ppo_mini_batches)


			for sample in data_generator:

				obs_batch, actions_batch, \
				   value_preds_batch, return_batch, old_action_log_probs_batch, \
						adv_targ,gt = sample

				values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
					obs_batch, actions_batch)

				ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
				surr1 = ratio * adv_targ
				surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
										   1.0 + self.clip_param) * adv_targ
				action_loss = -torch.min(surr1, surr2).mean()

				if self.use_clipped_value_loss:
					value_pred_clipped = value_preds_batch + \
						(values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
					value_losses = (values - return_batch).pow(2)
					value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
					value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
				else:
					value_loss = 0.5 * (return_batch - values).pow(2).mean()

				self.optimizer.zero_grad()


				# total_loss = (action_loss - dist_entropy * self.entropy_coef + value_loss*self.value_loss_coef)
				rl_loss = (action_loss - dist_entropy * self.entropy_coef + value_loss*self.value_loss_coef)

				if usenll:

					action = self.actor_critic.get_dec_sm(obs_batch)
					nll_loss = criterion(action,gt.squeeze().cuda().long())

					total_loss = 0.5*(rl_loss + nll_loss)
				else:
					total_loss = rl_loss

				total_loss.backward()

				nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)
				self.optimizer.step()

				value_loss_epoch += value_loss.item()
				action_loss_epoch += action_loss.item()
				dist_entropy_epoch += dist_entropy.item()
				total_loss_epoch += total_loss.item()

				if usenll:
					nll_loss_epoch += nll_loss.item()
				nll_loss_epoch = 0
				num_updates+=1


		# num_updates = self.ppo_epoch * self.batch_size

		value_loss_epoch /= num_updates
		action_loss_epoch /= num_updates
		dist_entropy_epoch /= num_updates
		nll_loss_epoch /= num_updates
		total_loss_epoch /= num_updates



		return value_loss_epoch, action_loss_epoch,dist_entropy_epoch,nll_loss_epoch,total_loss_epoch
