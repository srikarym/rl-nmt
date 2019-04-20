import gym
import gym_nmt
import torch
from baselines.common.vec_env import VecEnvWrapper
import torch.nn as nn
import torch
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env import CloudpickleWrapper
import itertools


def make_env(env_id, n_missing_words,seed,train_data,task):
    def _thunk():
        env = gym.make(env_id)
        env.init_words(n_missing_words,train_data,task)
        env.seed()

        return env

    return _thunk

def make_vec_envs(env_name,n_missing_words,seed,train_data,task,num_processes,train = True,num_sentences = 0,eval_env_name = 'nmt_eval-v0'):

    if train:
        envs = [make_env(env_name, n_missing_words ,seed+i,train_data,task)
                for i in range(num_processes)]
    else:
        envs = [make_env('nmt_eval-v0', n_missing_words,seed+i,train_data[i],task) \
                    for i in range(num_sentences)]

    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, 'cuda',task.source_dictionary.pad())

    return envs

def make_dummy(env_name,n_missing_words,train_data,task):

    dummy = gym.make(env_name)
    dummy.init_words(n_missing_words,train_data,task)
    return dummy

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device,pad):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.pad_val = pad


    def pad(self,obs):
        obs = list(map(list, zip(*obs)))
        source = obs[0]
        target = obs[1]

        max_size = 100
        
        sp = nn.utils.rnn.pad_sequence([torch.ones([max_size])] + [torch.tensor(s) for s in source] ,batch_first=True,padding_value=self.pad_val)

        tp = nn.utils.rnn.pad_sequence([torch.ones([max_size])] + [torch.tensor(s) for s in target] ,batch_first=True,padding_value=self.pad_val)
        return (sp[1:], tp[1:])

    def reset(self):
        
        obser = self.venv.reset()
        obs = []
        tac = []
        for ob in obser:
            obs.append(ob[0])
            tac.append(ob[1])
        return self.pad(obs),np.array(tac)

    def transition(self):
        self.venv.transition()


    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, tac = self.venv.step_wait()

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return self.pad(obs), reward, done,np.array(tac)



def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, tac = env.step(data)
                if done:
                    ob,tac = env.reset()
                remote.send((ob, reward, done, tac))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))

            elif cmd == 'transition':
                env.transition(words)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        self.specs = [f().spec for f in env_fns]
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def transition(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('transition',None))

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"


def _flatten_obs(obs):
    assert isinstance(obs, list) or isinstance(obs, tuple)
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        import collections
        assert isinstance(obs, collections.OrderedDict)
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return obs