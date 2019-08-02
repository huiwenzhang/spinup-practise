import numpy as np
import tensorflow as tf
import gym
import time
from first_round.algos.td3 import core
from first_round.algos.td3.core import get_vars
from first_round.utils.logx import EpochLogger


class ReplayBuffer:
    """
    FIFO experience reaplay buffer
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, obs_, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs_
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""
TD3 twin delayed DDPG
"""


def td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    :param env_fn:
    :param actor_critic:
    :param ac_kwargs:
    :param seed:
    :param steps_per_epoch:
    :param epochs:
    :param replay_size:
    :param gamma:
    :param polyak:
    :param pi_lr:
    :param q_lr:
    :param batch_size:
    :param start_steps:
    :param act_noise:
    :param target_noise:
    :param noise_clip:
    :param policy_delay:
    :param max_ep_len:
    :param logger_kwargs:
    :param save_freq:
    :return:
    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # we only used index 0 to assume all dimensions share the same limits
    act_limit = env.action_space.high[0]

    # feed env space info to actor-critic structure
    ac_kwargs['action_space'] = env.action_space

    # Placeholders
    x_ph, a_ph, r_ph, x2_ph, d_ph = core.placeholders(obs_dim, act_dim, None, obs_dim, None)

    # Main computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)

    with tf.variable_scope('target'):
        pi_targ, _, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)

    with tf.variable_scope('target', reuse=True):

        # Trick 1: add clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q values
        _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

    # Experience
    buf = ReplayBuffer(obs_dim, act_dim, replay_size)

    # Count variables
    var_counts = type(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

    # Bellman backup
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)

    # Losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1_targ - q1) ** 2)
    q2_loss = tf.reduce_mean((q2_targ - q2) ** 2)
    q_loss = q1_loss + q2_loss

    # Training operations
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging
    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                          outputs={'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or ep_len == max_ep_len):
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_len += 1
                ep_ret += r
            logger.store(TestEpRet=ep_ret, TestPeLen=ep_len)

    start = time.time()
    o, r, d, ep_len, ep_ret = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    for t in range(total_steps):
        if t > start_steps:
            a = get_action(0, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_len += 1
        ep_ret += r

        d = False if ep_len == max_ep_len else d

        # Store experience
        buf.store(o, a, r, o2, d)
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform updates at the end of a trajectory or when maximum steps reach"""
            for j in range(ep_len):
                # update at every time step
                batch = buf.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']}
                q_steps_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_steps_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                # Trick 3: deplayed policy update
                if j % policy_delay == 0:
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, re_len, ep_ret = env.reset(), 0, False, 0, 0

        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            test_agent()

            # Logger at the end of epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start)
            logger.dump_tabular()

    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='Halfcheetah-v2')
        parser.add_argument('--hid', type=int, default=300)
        parser.add_argument('--l', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--epoochs', type=int, default=50)
        parser.add_argument('--exp_name', type=str, default='td3')
        args = parser.parse_args()

        from first_round.utils.run_utils import setup_logger_kwargs
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

        td3(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
            ac_kwargs=dict(hid_size=[args.hid] * args.l), gamma=args.gamma,
            seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs)
