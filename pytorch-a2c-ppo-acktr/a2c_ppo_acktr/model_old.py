import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init
SOS_token = 0
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
		super(Policy, self).__init__()
		if base == 'Attn':
			base = AttnBase
			self.base = base(obs_shape[0],input_nwords = base_kwargs['input_nwords'],output_nwords=base_kwargs['output_nwords'],max_length=base_kwargs['max_length'],recurrent=True)
		else:

			if base_kwargs is None:
				base_kwargs = {}
			if base is None:
				if len(obs_shape) == 3:
					base = CNNBase
				elif len(obs_shape) == 1:
					base = MLPBase
				else:
					raise NotImplementedError

			self.base = base(obs_shape[0], **base_kwargs)

		if action_space.__class__.__name__ == "Discrete":
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "Box":
			num_outputs = action_space.shape[0]
			self.dist = DiagGaussian(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "MultiBinary":
			num_outputs = action_space.shape[0]
			self.dist = Bernoulli(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError
		self.dist.to(device)

	@property
	def is_recurrent(self):
		return self.base.is_recurrent

	@property
	def recurrent_hidden_state_size(self):
		"""Size of rnn_hx."""
		return self.base.recurrent_hidden_state_size

	def forward(self, inputs, rnn_hxs, masks):
		raise NotImplementedError


	def act(self, inputs, rnn_hxs, masks,first_time = False,context=None, deterministic=False):
		value, actor_features, rnn_hxs,context = self.base(inputs, rnn_hxs, masks,first_time = first_time,context = context)
		dist = self.dist(actor_features)

		# print('value is',value)
		# print('actor_features are',actor_features)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()
		# for f in self.dist.parameters():
		# 	print('data is',f.data)
		# 	print('grad is',f.grad)
		action_log_probs = dist.log_probs(action)
		# print('action probs are',dist.probs)

		return value, action, action_log_probs, rnn_hxs,context


	def get_value(self, inputs, rnn_hxs, masks):
		value, _, _,_ = self.base(inputs, rnn_hxs, masks)
		return value

	def evaluate_actions(self, inputs, rnn_hxs, masks, action):
		value, actor_features, rnn_hxs,_ = self.base(inputs, rnn_hxs, masks,first_time = True)
		

		dist = self.dist(actor_features)
		# print(self.dist)
		# for f in self.dist.parameters():
		# 	print('data is',f.data)
		# 	print('grad is',f.grad)
		# action_sample = dist.sample()

		# print('actions are',action)


		action_log_probs = dist.log_probs(action)
		# print('Probabilities are ',dist.probs)


		# print('action log probs are',action_log_probs)

		dist_entropy = dist.entropy().mean()

		# emp_entropy = dist.log_probs(action_sample)
		# action.detach()

		return value, action_log_probs, dist_entropy, rnn_hxs

class NNBase(nn.Module):

	def __init__(self, recurrent, recurrent_input_size, hidden_size):
		super(NNBase, self).__init__()

		self._hidden_size = hidden_size
		self._recurrent = recurrent

		if recurrent:
			self.gru = nn.GRU(recurrent_input_size, hidden_size).to(device)
			for name, param in self.gru.named_parameters():
				if 'bias' in name:
					nn.init.constant_(param, 0)
				elif 'weight' in name:
					nn.init.orthogonal_(param)
			torch.nn.utils.clip_grad_norm_(self.gru.parameters(), 0.25)

	@property
	def is_recurrent(self):
		return self._recurrent

	@property
	def recurrent_hidden_state_size(self):
		if self._recurrent:
			return self._hidden_size
		return 1

	@property
	def output_size(self):
		return self._hidden_size

	def _forward_gru(self, x, hxs, masks):
		if x.size(0) == hxs.size(0):

			x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
			x = x.squeeze(0)
			hxs = hxs.squeeze(0)
		else:
			# x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
			N = hxs.size(0)
			T = int(x.size(0) / N)

			# unflatten
			x = x.view(T, N, x.size(1))

			# Same deal with masks
			masks = masks.view(T, N)

			# Let's figure out which steps in the sequence have a zero for any agent
			# We will always assume t=0 has a zero in it as that makes the logic cleaner
			has_zeros = ((masks[1:] == 0.0) \
							.any(dim=-1)
							.nonzero()
							.squeeze()
							.cpu())


			# +1 to correct the masks[1:]
			if has_zeros.dim() == 0:
				# Deal with scalar
				has_zeros = [has_zeros.item() + 1]
			else:
				has_zeros = (has_zeros + 1).numpy().tolist()

			# add t=0 and t=T to the list
			has_zeros = [0] + has_zeros + [T]


			hxs = hxs.unsqueeze(0)
			outputs = []
			for i in range(len(has_zeros) - 1):
				# We can now process steps that don't have any zeros in masks together!
				# This is much faster
				start_idx = has_zeros[i]
				end_idx = has_zeros[i + 1]

				rnn_scores, hxs = self.gru(
					x[start_idx:end_idx],
					hxs * masks[start_idx].view(1, -1, 1)
				)

				outputs.append(rnn_scores)

			# assert len(outputs) == T
			# x is a (T, N, -1) tensor
			x = torch.cat(outputs, dim=0)
			# flatten
			x = x.view(T * N, -1)
			hxs = hxs.squeeze(0)

		return x, hxs


class CNNBase(NNBase):
	def __init__(self, num_inputs, recurrent=False, hidden_size=512):
		super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu'))

		self.main = nn.Sequential(
			init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
			nn.ReLU(),
			init_(nn.Conv2d(32, 64, 4, stride=2)),
			nn.ReLU(),
			init_(nn.Conv2d(64, 32, 3, stride=1)),
			nn.ReLU(),
			Flatten(),
			init_(nn.Linear(32 * 7 * 7, hidden_size)),
			nn.ReLU()
		)

		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(hidden_size, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, masks):
		x = self.main(inputs / 255.0)

		if self.is_recurrent:
			x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

		return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
	def __init__(self, num_inputs, recurrent=False, hidden_size=64):
		super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

		if recurrent:
			num_inputs = hidden_size

		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			np.sqrt(2))

		self.actor = nn.Sequential(
			init_(nn.Linear(num_inputs, hidden_size)),
			nn.Tanh(),
			init_(nn.Linear(hidden_size, hidden_size)),
			nn.Tanh()
		)

		self.critic = nn.Sequential(
			init_(nn.Linear(num_inputs, hidden_size)),
			nn.Tanh(),
			init_(nn.Linear(hidden_size, hidden_size)),
			nn.Tanh()
		)

		self.critic_linear = init_(nn.Linear(hidden_size, 1))

		self.train()

	def forward(self, inputs, rnn_hxs, masks):
		x = inputs

		if self.is_recurrent:
			x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

		hidden_critic = self.critic(x)
		hidden_actor = self.actor(x)

		return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden.float().to(device))
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)
class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size,max_length, dropout_p=0.1 ):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		# self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		# self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		# embedded = self.dropout(embedded) 

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								 encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnBase(NNBase):
	def __init__(self, num_inputs,input_nwords,output_nwords,max_length, recurrent=False, hidden_size=256):
		super(AttnBase, self).__init__(recurrent, 1,hidden_size)

		self.encoder = EncoderRNN(input_nwords, hidden_size).to(device)
		self.decoder = AttnDecoderRNN(hidden_size, output_nwords,max_length, dropout_p=0.1).to(device)
		init_ = lambda m: init(m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0))
		self.critic_linear = init_(nn.Linear(hidden_size, 1)).to(device)

		self.train()
		self.max_length = max_length

	def forward(self,inputs,rnn_hxs,masks,first_time,context=None):
		source = inputs[0][0]
		targets = inputs[0][1]

		inp_index = (source == -1).nonzero()[0].cpu().numpy()[0]
		tar_index = (targets == -1).nonzero()[0].cpu().numpy()[0]

		# target_gen = torch.tensor(target_n).to(device)

		input_tensor = source[:inp_index].long().cuda()
		target_gen =  targets[:tar_index].long().cuda()



		if (first_time):
			encoder_hidden = self.encoder.initHidden()

			# input_tensor = torch.tensor(source).to(device)

			input_length = input_tensor.size(0)
			target_length = target_gen.size(0)

			encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)
			for ei in range(input_length):
				encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] = encoder_output[0, 0]
		
			decoder_input = torch.tensor([ [SOS_token] ], device=device)
			decoder_hidden = encoder_hidden


			for di in range(target_length):
				_, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				decoder_input = target_gen[di]
			decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				
		else:
			decoder_hidden = context[0]
			encoder_outputs = context[1]
			target_length = target_gen.size(0)
			if target_length == 0:
				decoder_input = torch.tensor([[SOS_token]], device=device)
			else:
				decoder_input = target_gen[-1]
			decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)



		topv, topi = decoder_output.topk(1)

		# decoder_output = topi.squeeze().detach()

		rnn_hxs = rnn_hxs.to(device)
		masks = masks.to(device)
		x = topi.float().to(device)


		if self.is_recurrent:
			x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
		# print('x is',x)
		# print('rnn_hxs',rnn_hxs)


		# for f in self.gru.parameters():
		# 	print('data is',f.data)
		# 	print('grad is',f.grad)
		return self.critic_linear(x), x, rnn_hxs,[decoder_hidden,encoder_outputs]


