import time

import gym
import numpy as np
import tensorflow as tf

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
        self.logp_buf = np.zeros(size, dtype=np.float32)
        print(info_shapes)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k, v in info_shapes.items()}
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
        this function is called at the end of an episode, if the episode is cut off
        the final reward should be: R(s) = r + v(s_next). Otherwise, it is r
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

    def get(self):
        """
        return all the samples in this epoch and ten start training
        :return:
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # advantage normalization
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf] + core.values_as_sorted_list(self.info_bufs)


"""
TRPO
"""


def trpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lf=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo'):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # shared ac_kwargs
    ac_kwargs['action_space'] = env.action_space

    # placeholders
    x_ph, a_ph = core.placeholder_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # network and output
    # info is distribution dependant, if it is continuous, info is mu and log_std
    # info_phs = {'mu_old': mu, 'old_log_std': log_std}
    pi, logp, logp_pi, info, info_phs, d_kl, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # order placeholders
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph] + core.values_as_sorted_list(info_phs)

    # action operations
    get_action_ops = [pi, v, logp_pi] + core.values_as_sorted_list(info)

    # expriment buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    info_shapes = {k: v.shape.as_list()[1:] for k, v in info_phs.items()}
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    # count variable
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # TRPO losses
    ratio = tf.exp(logp - logp_old_ph)
    pi_loss = -tf.reduce_mean(ratio * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # optimizer for value function
    train_vf = MpiAdamOptimizer(learning_rate=vf_lf).minimize(v_loss)

    # optimizer about pi
    pi_params = core.get_vars('pi')
    gradient = core.flat_grad(pi_loss, pi_params)  # gradient of pi
    # define the placeholder and Hessian matrix
    # Hx = g, so x = H^-1 g, here hvp = Hx
    v_ph, hvp = core.hessian_vector_product(d_kl, pi_params)
    if damping_coeff > 0:
        hvp += damping_coeff * v_ph

    get_pi_params = core.flat_concat(pi_params)
    set_pi_params = core.assign_params_from_flat(v_ph, pi_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # sync params
    sess.run(sync_all_params())

    # model saveing setup
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    # utilities for updating
    def cg(Ax, b):
        """
        conjugate gradient algorithm
        :param Ax:
        :param b:
        :return:
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new

        return x

    def update():
        # parepare Hessian and gradient
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        Hx = lambda x: mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

        # mian computation
        x = cg(Hx, g)  # compute x = H^-1 g
        alpha = np.sqrt(2 * delta / (np.dot(x, Hx(x)) + EPS))  # theta = theta + alpha * x
        old_params = sess.run(get_pi_params)

        def set_and_eval(step):
            # update weights with natural gradient,
            # TODO: why it is minus here
            sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
            return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))

        if algo == 'npg':
            # npg has no backtracking -- using hard kl constraint
            kl, pi_l_new = set_and_eval(step=1.)
        elif algo == 'trpo':
            # use line search to make sure kl constraint is valid
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepte new parameters at %d of line search.' % j)
                    logger.store(BacktrackIters=j)
                    break

                if j == backtrack_iters - 1:
                    logger.log('Line search failed, keep old params')
                    logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(0.)  # step = 0 means no update

        # value function update
        for _ in range(train_v_iters):
            sess.run(train_vf, feed_dict=inputs)
        v_l_new = sess.run(v_loss, feed_dict=inputs)

        # log changes (save variables)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # MAIN LOOP
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            outs = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})
            a, v_t, logp_t, info_t = outs[0][0], outs[1], outs[2], outs[3:]
            buf.store(o, a, r, v_t, logp_t, info_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)  # terminal when episode is done
            if terminal or (t == local_steps_per_epoch):
                if not (terminal):
                    print('Waring: trajectory cut off by epoch at {} steps'.format(ep_len))
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_len, ep_ret = env.reset(), 0, False, 0, 0

        # save model at the end of an episode
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        update()

        # log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch - 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('KL', average_only=True)
        if algo == 'trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time() - start)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from first_round.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hid_size=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
