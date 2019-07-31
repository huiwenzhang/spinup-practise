import numpy as np
import tensorflow as tf


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mlp(x, hid_size=(32,), activ=tf.tanh, output_activ=None):
    for h in hid_size[:-1]:
        x = tf.layers.dense(x, h, activ)
    return tf.layers.dense(x, hid_size[-1], output_activ)


def get_vars(scope):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope):
    return sum([np.prod(var.shape.as_list()) for var in get_vars(scope)])


"""
Actor-Critic
"""


def mlp_actor_critic(x, a, hid_size=(400, 300), activ=tf.nn.relu, output_activ=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hid_size) + [act_dim], activ, output_activ)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hid_size) + [1], activ, None))
    with tf.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hid_size) + [1], activ, None))

    return pi, q, q_pi
