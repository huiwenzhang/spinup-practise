import numpy as np
import tensorflow as tf
import gym
import time
import first_round.algos.trpo.core as core
from first_round.utils.logx import EpochLogger
from first_round.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from first_round.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

EPS = 1e-8

class GAEBuffer:
    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        # we store logp because we want to esitmate kl between new pi and old_pi
        # we can't recompute an outdated pi, so store it
        self.logp_buf = np.zeros(size, dtype=np.float23)
        self.info_bufs = {k: np.zeros([size] + list[v], dtype=np.float32) for k, v in info_shapes}
        self.sorted_info_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size


    def store(self, obs, act, rew, val, logp, info):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        usually the last value contains two situations:
            - the episode is over, agent died by entering a terminal state
            - no terminal state, episode is over because of max length reached or we cut off
        :param last_val:
        :return:
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # reward-to-go
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

