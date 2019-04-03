import torch
import torch.nn as nn
import numpy as np
import fairseq
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init
import lstm
from fairseq import tasks
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sacrebleu import sentencebleu

class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
		super(Policy, self).__init__()
		if base == 'LSTM':
			base = LSTMBase
			self.base = base(base_kwargs['n_proc'], base_kwargs['dummyenv'])
			self.base.to(device)

		elif base == 'FConv':
			base = FConv
			self.base = base(base_kwargs['n_proc'], base_kwargs['dummyenv'])
			self.base.to(device)

		elif base == 'Transformer':
			base = TBase
			self.base = base(base_kwargs['task'], base_kwargs['args'])
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

	def bleuscore(self,data,dummyenv):

		avg_bleu = 0.0

		for d in data:
			source = d['net_input']['src_tokens']
			truetarget = d['target']

			max_len = int(len(truetarget)*1.5)
			currlen = 0

			generation = d['net_input']['prev_output_tokens']

			while (currlen < max_len):
				action = self.base((source,generation),None,True)
				generation = torch.cat((generation,action))

				if action[0].cpu().numpy()[0] == dummyenv.target_dictionary.eos():
					break

			hyp = dummyenv.target_dictionary.string(generation, bpe_symbol='@@ ').replace('@@ ','')
			ref = dummyenv.target_dictionary.string(truetarget, bpe_symbol='@@ ').replace('@@ ','')

			avg_bleu += sentencebleu(hyp,ref)

		return avg_bleu/len(data)



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



class LSTMBase(NNBase):
	def __init__(self, task, dummy_env,args, recurrent=False, hidden_size=256):
		super(LSTMBase, self).__init__(recurrent, hidden_size, 512)

		model = task.build_model(args)
		model.cuda()
		self.encoder = model.encoder
		self.decoder = model.decoder

		# self.encoder = fairseq.models.lstm.LSTMEncoder(self.dummyenv.task.source_dictionary, left_pad=False,
		#                                                num_layers=2, dropout_in=0.0, dropout_out=0.0,
		#                                                padding_value=1.0).to(device)
		# self.decoder = lstm.LSTMDecoder(self.dummyenv.task.target_dictionary, dropout_in=0.0, num_layers=2,
		#                                 dropout_out=0.0).to(device)

		init_ = lambda m: init(m,
							   nn.init.orthogonal_,
							   lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(256, 1)).to(device)
		self.pad_value = dummy_env.task.source_dictionary.pad()

		self.train()

	def forward(self, inputs, tac=None, generation = True):


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

		enc_out = self.encoder(s, torch.tensor(idx).long().to(device))

		dec_out,dec_hidden = self.decoder(t, enc_out)

		m = torch.nn.Softmax(dim=-1)

		if generation:
			return m(dec_out)

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


class TBase(NNBase):
	def __init__(self, task, args, recurrent=False, hidden_size=256):
		super(TBase, self).__init__(recurrent, hidden_size, 512)


		self.model = task.build_model(args).to(device)
		init_ = lambda m: init(m,
							   nn.init.orthogonal_,
							   lambda x: nn.init.constant_(x, 0))		

		self.critic_linear = init_(nn.Linear(512, 1)).to(device)
		self.pad_value = task.source_dictionary.pad()

		self.train()

	def forward(self, inputs, tac=None):

		s = inputs[0].long().to(device)
		t = inputs[1].long().to(device)

		nos = []
		for i in range(s.shape[0]):
			nos.append(int(torch.sum(s[i] == 1).cpu().numpy()))
			
		if (min(nos) != 0):
			s = s[:,- (100 - min(nos)):]

		lens = [100-n for n in nos]		

		obs = {'src_tokens':s,
					'src_lengths':torch.tensor(lens).long().to(device),
					'prev_output_tokens':t}

		dec_hidden,dec_out = self.model(**obs)

		outs = dec_out[:,-1,:]
		hidden = dec_hidden[:,-1,:]

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

class FConv(NNBase):
	def __init__(self, num_inputs, dummy_env, recurrent=False, hidden_size=256):
		super(FConv, self).__init__(recurrent, hidden_size, 512)

		self.dec_out_dim = 256 

		self.encoder = fairseq.models.fconv.FConvEncoder(dictionary=dummy_env.task.src_dict,embed_dim=256,\
						convolutions = eval('[(256, 3)] * 4'),left_pad = False,max_positions=100)
		self.decoder = fairseq.models.fconv.FConvDecoder(dictionary=dummy_env.task.tgt_dict,embed_dim=256,\
						convolutions=eval('[(256, 3)] * 3'),out_embed_dim=self.dec_out_dim)

		self.encoder.num_attention_layers = sum(layer is not None for layer in self.decoder.attention)


		init_ = lambda m: init(m,
							   nn.init.orthogonal_,
							   lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(self.dec_out_dim, 1)).to(device)
		self.pad_value = dummy_env.task.source_dictionary.pad()

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
		enc_out = self.encoder(s, torch.tensor(idx).long().to(device))

		dec_hidden,dec_out = self.decoder(t, enc_out)
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