import torch
import numpy as np

def _flatten_helper(_tensor):
    T,N = _tensor.size()[:2]
    return _tensor.view(T * N, -1)


class RolloutStorage(object):
    def __init__(self, num_steps, num_rolls_per_sen, num_processes, obs_shape, action_space):
        # self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.obs_s = torch.ones(num_steps*num_rolls_per_sen+1,num_processes,100)
        self.obs_t = torch.ones(num_steps*num_rolls_per_sen+1,num_processes,100)

        self.num_processes = num_processes
        self.num_steps = num_steps
        self.num_rolls_per_sen = num_rolls_per_sen
        # self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps * num_rolls_per_sen, num_processes, 1)
        self.value_preds = torch.zeros(num_steps * num_rolls_per_sen +1, num_processes, 1)
        self.returns = torch.zeros(num_steps*num_rolls_per_sen+1,num_processes,1)
        self.action_log_probs = torch.zeros(num_steps * num_rolls_per_sen, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps * num_rolls_per_sen, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps  * num_rolls_per_sen+1, num_processes, 1)


        self.step = 0
        self.roll = 0

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

    def insert(self,obs, actions, action_log_probs, value_preds, rewards, masks):

        self.obs_s[self.step + 1].copy_(obs[0])
        self.obs_t[self.step+1].copy_(obs[1])
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step+1].copy_(masks)

        self.step = (self.step + 1) % (self.num_steps * self.num_rolls_per_sen)


    def after_update(self):
        self.obs_s[0].copy_(self.obs_s[-1])
        self.obs_t[0].copy_(self.obs_t[-1])

        self.masks[0].copy_(self.masks[-1])

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

    def feed_forward_generator(self, advantages, batch_size):


        obs_s_flat = _flatten_helper(self.obs_s[:-1])
        obs_t_flat = _flatten_helper(self.obs_t[:-1])

        total = obs_s_flat.shape[0]
        arr = np.arange(total)
        np.random.shuffle(arr)
        indices = arr[:batch_size]


        obs_batch_s = obs_s_flat[indices]
        obs_batch_t = obs_t_flat[indices]

        actions_flat = self.actions.view(-1, 1)
        value_preds_flat = self.value_preds.view(-1, 1)
        returns_flat = self.returns.view(-1, 1)
        action_log_probs_flat = self.action_log_probs.view(-1, 1)
        advantages_flat = advantages.view(-1, 1)

        actions_batch = actions_flat[indices]
        value_preds_batch = value_preds_flat[indices]
        return_batch = returns_flat[indices]
        old_action_log_probs_batch = action_log_probs_flat[indices]
        adv_targ = advantages_flat[indices]

        return (obs_batch_s, obs_batch_t), actions_batch, \
               value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            # obs_batch = []
            obs_s_batch = []
            obs_t_batch = []

            actions_batch = []
            value_preds_batch = []
            return_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_s_batch.append(self.obs_s[:-1, ind])
                obs_t_batch.append(self.obs_t[:-1, ind])

                # recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                # masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            # obs_batch = torch.stack(obs_batch, 1)
            obs_s_batch = torch.stack(obs_s_batch, 1)
            obs_t_batch = torch.stack(obs_t_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            # masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            # recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            # obs_batch = _flatten_helper(T, N, obs_batch)
            obs_s_batch = _flatten_helper(T, N, obs_s_batch)
            obs_t_batch = _flatten_helper(T, N, obs_t_batch)

            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            # masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (obs_s_batch, obs_t_batch), actions_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
