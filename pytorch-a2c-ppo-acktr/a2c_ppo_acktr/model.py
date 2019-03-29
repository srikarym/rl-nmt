import torch
import torch.nn as nn
import numpy as np
import fairseq
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init
# import lstm
from fairseq import tasks

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

task_args = AttrDict()

task_args.arch='lstm' 

task_args.encoder_layers=2
task_args.decoder_layers=2

task_args.criterion='label_smoothed_cross_entropy'
task_args.data=['data/iwslt14.tokenized.de-en']
task_args.left_pad_source='False'
task_args.left_pad_target='False'
task_args.source_lang='de'
task_args.target_lang='en'
task_args.task='translation'


task = tasks.setup_task(task_args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
		super(Policy, self).__init__()
		if base == 'Attn':
			base = AttnBase
			self.base = base(base_kwargs['n_proc'], base_kwargs['dummyenv'], recurrent=base_kwargs['recurrent'])
			self.base.to(device)
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

	def act(self, inputs, tac, deterministic=False):
		value, actor_features, ranks = self.base(inputs, tac)

		dist = self.dist(actor_features)

		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)

		return value,action,action_log_probs,ranks

	def get_value(self, inputs):
		value, _, _ = self.base(inputs)
		return value

	def evaluate_actions(self, inputs, action):
		value, actor_features, _ = self.base(inputs)

		dist = self.dist(actor_features)

		action_log_probs = dist.log_probs(action)

		dist_entropy = dist.entropy().mean()

		return value,action_log_probs,dist_entropy


class NNBase(nn.Module):

	def __init__(self, recurrent, recurrent_input_size, hidden_size):
		super(NNBase, self).__init__()

		self._hidden_size = hidden_size
		self._recurrent = recurrent


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



class AttnBase(NNBase):
	def __init__(self, num_inputs, dummy_env, recurrent=False, hidden_size=256):
		super(AttnBase, self).__init__(recurrent, hidden_size, 512)
		self.num_inputs = num_inputs
		self.dummyenv = dummy_env

		# self.encoder = fairseq.models.lstm.LSTMEncoder(self.dummyenv.task.source_dictionary, left_pad=False,
		#                                                num_layers=2, dropout_in=0.0, dropout_out=0.0,
		#                                                padding_value=1.0).to(device)
		# self.decoder = lstm.LSTMDecoder(self.dummyenv.task.target_dictionary, dropout_in=0.0, num_layers=2,
		#                                 dropout_out=0.0).to(device)

		self.model = task.build_model(task_args).cuda()


		init_ = lambda m: init(m,
							   nn.init.orthogonal_,
							   lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(512, 1)).to(device)
		self.pad_value = self.dummyenv.task.source_dictionary.pad()

		self.train()

	def forward(self, inputs, tac=None):
		s = inputs[0].long().to(device)
		t = inputs[1].long().to(device)

		nos = []
		for i in range(s.shape[0]):
			nos.append(int(torch.sum(s[i] == self.pad_value).cpu().numpy()))


		if (min(nos) != 0):
			s = s[:, :s.shape[1] - min(nos)]

		idx = []
		for i in range(s.shape[0]):
			l = (s[i] == self.pad_value).nonzero()
			if l.shape[0] == 0:
				l = s.shape[1]
			else:
				l = l[0].cpu().numpy()[0]
			idx.append(l)

		obs = {'src_tokens':s,
					'src_lengths':torch.tensor(idx).long().to(device),
					'prev_output_tokens':t}

		# enc_out = self.encoder(s, torch.tensor(idx).long().to(device))

		# dec_hidden,dec_out = self.decoder(t, enc_out)

		dec_out,dec_hidden = self.model(**obs)

		idx = []
		for i in range(t.shape[0]):
			l = (t[i] == self.pad_value).nonzero()
			if l.shape[0] == 0:
				l = -1
			else:
				l = l[0]-1
			idx.append(l)

		outs = dec_out[np.arange(dec_out.shape[0]), idx, :]
		hidden = dec_hidden[np.arange(dec_hidden.shape[0]), idx, :]

		m = torch.nn.Softmax(dim=-1)
		sm = m(outs)


		if tac is None:
			return self.critic_linear(hidden), sm, None
		else:

			sm_np = sm.cpu().numpy()
			probs = sm_np[np.arange(sm_np.shape[0]),tac]

			np.ndarray.sort(sm_np)

			ranks = []
			for i in range(t.shape[0]):
				for j in range(sm_np[0].shape[0]):
					if sm_np[i][j] == probs[i]:
						break
				ranks.append(sm_np[0].shape[0] - j)


			return self.critic_linear(hidden), sm, ranks