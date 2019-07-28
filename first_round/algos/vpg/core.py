import numpy as np
import scipy.signal
import tensorflow as tf
from gym.spaces import Box, Discrete

EPS = 1e-8


def combined_shape(length, shape=None):
    """
    combine shape of lenght and shape, return a tuple
    :param length:
    :param shape:
    :return:
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    """
    create a tf placehold with the given dimension
    :param dim:
    :return:
    """
    return tf.placeholder(tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    """
    create multiple placeholders for different variables, such as obs, rew,
    :param args:
    :return:
    """
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        # if space is continous, the shape is the dimension of the action
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        # if actions is discrete, it must be scalar
        return tf.placeholder(tf.int32, shape=(None,))
    else:
        raise NotImplementedError


def placeholder_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hid_size=(32,), activ=tf.tanh, output_activ=None):
    """
    create a mlp network with given hidden layer size and activation function
    :param x:
    :param hid_size:
    :param activ:
    :param output_activ:
    :return:
    """
    for layer in hid_size[:-1]:
        x = tf.layers.dense(x, layer, activ)
    return tf.layers.dense(x, hid_size[-1], activation=output_activ)


def get_vars(scope=''):
    """
    get trainable variables in specific scope, used to identify policy or critic net weights
    :param scope:
    :return:
    """
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    """
    return log p(x) for given x when p obeys Gaussian distribution with mean mu
    and log std as log_std. Note we use log_std
    :param x:
    :param mu:
    :param log_std:
    :return:
    """
    # solved by apply log operation to Gaussian density function
    prob = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_mean(prob, axis=1)


def discount_cumsum(x, discount):
    """
    compute discounted cumulative reward from an episode reward
    :param x: input, for rl it is a sequence of reward
    :param discount:
    :return:
     input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]

      we can compute this number by first reverse the x, then process backward with rule:
      x_t = x_t + discount * x_{t+1}
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
policies:
"""


def mlp_categorical_policy(x, a, hid_size, activ, output_activ, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hid_size) + [act_dim], activ, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    # get log prob for given act a
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hid_size, activ, output_activ, action_space):
    act_dim = a.shape.as_list()[-1]  # dimension of action
    mu = mlp(x, list(hid_size) + [act_dim], activ, output_activ)
    # log_std is not a matrix here
    log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


""""
Actor-Critic
"""


def mlp_actor_critic(x, a, hid_size=(64, 64), activ=tf.tanh, output_activ=None, policy=None,
                     action_space=None):
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hid_size, activ, output_activ, action_space)
    with tf.variable_scope('v'):
        # output value is a scalar, so we append 1 as the last number of unit, pay attention: we
        # need to squeeze v to keep the shape is (batch,) instead of (batch, 1)
        v = tf.squeeze(mlp(x, list(hid_size) + [1], activ, None), axis=1)
    return pi, logp, logp_pi, v
