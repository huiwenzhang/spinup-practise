import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    else:
        raise NotImplementedError


def placeholder_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hid_size=(32,), activ=tf.tanh, output_activ=None):
    for h in hid_size[:-1]:
        x = tf.layers.dense(x, h, activ)
    return tf.layers.dense(x, hid_size[-1], output_activ)


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""


def mlp_categorical_policy(x, a, hid_size, activ, output_activ, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hid_size) + [act_dim], activ, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hid_size, activ, output_activ, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hid_size) + [act_dim], activ, output_activ)
    log_std = tf.get_variable('log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    # logp use action placeholder as input, so it is old, while logp_pi is new
    return pi, logp, logp_pi


"""
Actor-Critic
"""


def mlp_actor_critic(x, a, hid_size=(64, 64), activ=tf.tanh, output_activ=None, policy=None, action_space=None):
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hid_size, activ, output_activ, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hid_size) + [1], activ, None), axis=1)
    return pi, logp, logp_pi, v
