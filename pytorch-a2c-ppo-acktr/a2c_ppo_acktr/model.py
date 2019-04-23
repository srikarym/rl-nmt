import torch
import torch.nn as nn
import numpy as np
import fairseq
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init
# import lstm
from fairseq import tasks
from fairseq.utils import import_user_module
from fairseq.utils import _upgrade_state_dict
from fairseq import bleu, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from misc import args, SacrebleuScorer
from copy import deepcopy

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



class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
		super(Policy, self).__init__()
		if base == 'Attn':
			base = AttnBase
			self.base = base(base_kwargs['n_proc'], base_kwargs['dummyenv'], recurrent=base_kwargs['recurrent'])
			self.base.to(device)


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

		if args.max_tokens is None and args.max_sentences is None:
			args.max_tokens = 12000

		self.align_dict = utils.load_align_dict(args.replace_unk)
		self.task = tasks.setup_task(args)
		self.task.load_dataset(args.gen_subset)
		self.src_dict = getattr(self.task, 'source_dictionary', None)
		self.tgt_dict = self.task.target_dictionary
		self.generator = self.task.build_generator(args)


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

	def bleuscore(self):

		model = self.base.model
		# print(model)

		itr = self.task.get_batch_iterator(
			dataset=self.task.dataset(args.gen_subset),
			max_tokens=args.max_tokens,
			max_sentences=args.max_sentences,
			max_positions=utils.resolve_max_positions(
				self.task.max_positions(),
				*[model.max_positions() ]
			),
			ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
			required_batch_size_multiple=args.required_batch_size_multiple,
			num_shards=args.num_shards,
			shard_id=args.shard_id,
			num_workers=args.num_workers,
		).next_epoch_itr(shuffle=False)

		scorer = SacrebleuScorer()
		use_cuda = torch.cuda.is_available() and not args.cpu
		has_target = True


		with progress_bar.build_progress_bar(args, itr) as t:
			for sample in t:
				sample = utils.move_to_cuda(sample) if use_cuda else sample
				if 'net_input' not in sample:
					continue

				prefix_tokens = None
				if args.prefix_size > 0:
					prefix_tokens = sample['target'][:, :args.prefix_size]

				hypos = self.task.inference_step(self.generator, [model], sample, prefix_tokens)
				num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)

				for i, sample_id in enumerate(sample['id'].tolist()):
					has_target = sample['target'] is not None

					# Remove padding
					src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], self.tgt_dict.pad())
					target_tokens = None
					if has_target:
						target_tokens = utils.strip_pad(sample['target'][i, :], self.tgt_dict.pad()).int().cpu()

					# Either retrieve the original sentences or regenerate them from tokens.
					if self.align_dict is not None:
						src_str = self.task.dataset(args.gen_subset).src.get_original_text(sample_id)
						target_str = self.task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
					else:
						if self.src_dict is not None:
							src_str = self.src_dict.string(src_tokens, args.remove_bpe)
						else:
							src_str = ""
						if has_target:
							target_str = self.tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)


					# Process top predictions
					for i, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
						hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
							hypo_tokens=hypo['tokens'].int().cpu(),
							src_str=src_str,
							alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
							align_dict=self.align_dict,
							tgt_dict=self.tgt_dict,
							remove_bpe=args.remove_bpe,
						)


						# Score only the top hypothesis
						if has_target and i == 0:
							if self.align_dict is not None or args.remove_bpe is not None:
								# Convert back to tokens for evaluation with unk replacement and/or without BPE
								target_tokens = self.tgt_dict.encode_line(target_str, add_if_not_exist=True)
							if hasattr(scorer, 'add_string'):
								scorer.add_string(target_str, hypo_str)
							else:
								scorer.add(target_tokens, hypo_tokens)


		return scorer.corpus_bleu(), scorer.sentence_bleu()

	def get_dec_sm(self,inputs):
		action = self.base(inputs,None,False,True)
		return action



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


		self.model = task.build_model(task_args).cuda()


		init_ = lambda m: init(m,
							   nn.init.orthogonal_,
							   lambda x: nn.init.constant_(x, 0))

		self.critic_linear = init_(nn.Linear(512, 1)).to(device)
		self.pad_value = self.dummyenv.task.source_dictionary.pad()

		self.train()

	def forward(self, inputs, tac=None,generation = False,logsoftmax = False):
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

		# obs = {'src_tokens':s,
		# 			'src_lengths':torch.tensor(idx).long().to(device),
		# 			'prev_output_tokens':t}

		# dec_out,dec_hidden = self.model(**obs)

		enc_out = self.model.encoder(s, torch.tensor(idx).long().to(device))

		dec_out,dec_hidden = self.model.decoder(t, enc_out,None,True)

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

		if logsoftmax:
			m = nn.LogSoftmax()
			return m(outs)

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
