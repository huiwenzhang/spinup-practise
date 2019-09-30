import numpy as np
import tensorflow as tf

EPS = 1e-8


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [tf.placeholder(dim) for dim in args]


def mlp(x, hid_size=(32,), activ=tf.tanh, output_activ=None):
    for h in hid_size[:-1]:
        x = tf.layers.dense(x, h, activ)
    return tf.layers.dense(x, hid_size[-1], output_activ)


def get_vars(scope):
    return [var for var in tf.trainable_variables() if scope in var.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre, axis=1)


def clip_but_pass_gradient(x, low=-1, high=1.):
    # this implementation achieves the similar functionality as tf.clip_by_value
    clip_up = tf.cast(x > high, tf.float32)
    clip_low = tf.case(x < low, tf.float32)
    return x + tf.stop_gradient((high - x) * clip_up + (low - x) * clip_low)


""""
Policies
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp_gaussian_policy(x, a, hid_size, activ, output_activ):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hid_size), activ, activ)
    mu = tf.layers.dense(net, act_dim, output_activ)

    # log_std: a function  of state
    # using tanh activation to clip the std in a range
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    # since log_std in (-1, 1), log_std+1 in (0, 2), thus the flowing range is (LOG_STD_MIN, LOG_STD_MAX)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)  # standard variance to variance
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi


def apply_squashing_func(mu, pi, logp_pi):
    # why scale here
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # add 1e-6 to make sure the smalleset log value is bigger than loge-6 = -6
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi ** 2, low=0, high=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-critic
"""


def mlp_actor_critic(x, a, hid_size=(400, 300), activ=tf.relu, output_activ=None, policy=mlp_gaussian_policy,
                     action_space=None):
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = policy(x, a, hid_size, activ, output_activ)
        # squash
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        action_scale = action_space.high[0]
        mu *= action_scale  # scale back
        pi *= action_scale

        # value function, vf_mlp is a function of x
        vf_mlp = lambda x: tf.squeeze(mlp(x, list(hid_size) + [1], activ, None), axis=1)
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([x, a], axis=-1))
        with tf.variable_scope('q1', reuse=True):
            q1_pi = vf_mlp(tf.concat([x, pi], axis=-1))
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([x, a], axis=-1))
        with tf.variable_scope('q2', reuse=True):
            q2_pi = vf_mlp(tf.concat([x, pi], axis=-1))
        with tf.variable_scope('v'):
            v = vf_mlp(x)
        return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v
