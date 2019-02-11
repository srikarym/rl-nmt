import torch
import torch.nn as nn
import numpy as np
import fairseq
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init
import lstm

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

        # dist = torch.distributions.Categorical(probs=actor_features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # print(action.shape)

        # action_log_probs = dist.log_probs(action.squeeze(-1))
        action_log_probs = dist.log_probs(action)

        # return value, action, action_log_probs.unsqueeze(1), ranks
        return value,action,action_log_probs,ranks

    def get_value(self, inputs):
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features, _ = self.base(inputs)

        # dist = torch.distributions.Categorical(probs=actor_features)
        dist = self.dist(actor_features)

        # action_log_probs = dist.log_prob(action.squeeze(-1))
        action_log_probs = dist.log_probs(action)


        dist_entropy = dist.entropy().mean()

        # return value, action_log_probs.unsqueeze(1), dist_entropy
        return value,action_log_probs,dist_entropy


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

class AttnBase(NNBase):
    def __init__(self, num_inputs, dummy_env, recurrent=False, hidden_size=256):
        super(AttnBase, self).__init__(recurrent, hidden_size, 512)
        self.num_inputs = num_inputs
        self.dummyenv = dummy_env

        self.encoder = fairseq.models.lstm.LSTMEncoder(self.dummyenv.task.source_dictionary, left_pad=False,
                                                       num_layers=2, dropout_in=0.0, dropout_out=0.0,
                                                       padding_value=1.0).to(device)
        self.decoder = lstm.LSTMDecoder(self.dummyenv.task.target_dictionary, dropout_in=0.0, num_layers=2,
                                        dropout_out=0.0).to(device)

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

        # if self.is_recurrent:
        #   outs, rnn_hxs = self._forward_gru(outs, rnn_hxs, masks)

        if tac is None:
            return self.critic_linear(hidden), sm, None
        else:

            sm_np = sm.cpu().numpy()
            probs = sm_np[np.arange(sm_np.shape[0]),tac]

            np.ndarray.sort(sm_np)

            ranks = []
            for i in range(t.shape[0]):
                # print(np.nonzero(sm_np[i] == probs[i]))
                for j in range(sm_np[0].shape[0]):
                    if sm_np[i][j] == probs[i]:
                        a = True
                        break
                ranks.append(sm_np[0].shape[0] - j)


            return self.critic_linear(hidden), sm, ranks