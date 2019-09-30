import numpy as np
import scipy.signal
import tensorflow as tf
from gym.spaces import Box, Discrete

EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    """
    return the value of a dict sorted by its keys
    :param dict:
    :return:
    """
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
    for layer in hid_size[:-1]:
        x = tf.layers.dense(x, layer, activ)
    return tf.layers.dense(x, hid_size[-1], activation=output_activ)


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    """
    compute loglikelihood for input x given Gaussian distribution with mean mu and
    logged std log_std
    :param x:
    :param mu: list, mean value
    :param log_std: list, each entry of the list is a number of the diagonal entry of covariance matrix
    :return:
    """
    prob = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(prob, axis=1)


def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    # kl divergence between two Gaussians: kl(q0 || q1)
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)  # var = std **2
    kl = 0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
    all_kls = tf.reduce_sum(kl, axis=1)
    return tf.reduce_mean(all_kls)


def categoriacl_kl(logp0, logp1):
    # we actually compute kl(p1 || p0)
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    # why use mean value here
    return tf.reduce_mean(all_kls)


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)  # column vector


def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))


def hessian_vector_product(f, params):
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g * x), params)


def assign_params_from_flat(x, params):
    """
    assign value of params to x
    :param x:
    :param params:
    :return:
    """
    flat_size = lambda p: int(np.prod(p.shape.as_list()))
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])


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

    old_logp_all = placeholder(act_dim)
    d_kl = categoriacl_kl(logp_all, old_logp_all)  # kl(old || new)

    info = {'logp_all': logp_all}
    info_phs = {'logp_all': old_logp_all}

    return pi, logp, logp_pi, info, info_phs, d_kl


def mlp_gaussian_policy(x, a, hid_size, activ, output_activ, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hid_size) + [act_dim], activ, output_activ)
    log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(shape=tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
    d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)  # kl(new || old)

    info = {'mu': mu, 'log_std': log_std}
    info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, info, info_phs, d_kl


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
        policy_outs = policy(x, a, hid_size, activ, output_activ, action_space)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hid_size) + [1], activ, None), axis=1)
    return pi, logp, logp_pi, info, info_phs, d_kl, v
