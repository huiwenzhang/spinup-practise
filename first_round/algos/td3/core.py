import numpy as np
import tensorflow as tf


def placeholder(dim):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [tf.placeholder(dim) for dim in args]


def mlp(x, hid_size=(32,), activ=tf.tanh, output_activ=None):
    for h in hid_size[:-1]:
        x = tf.layers.dense(x, h, activ)
    return tf.layers.dense(x, hid_size[-1], output_activ)


def get_vars(scope):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


"""
Actor-Critic
"""


def mlp_actor_critic(x, a, hid_size=(400, 300), activ=tf.nn.relu, output_activ=tf.tanh,
                     action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        # We use relu activation for hidden layer, use tanh activation for the output
        # layer to clip the output in range [-1, 1] and scaled by act_limit
        pi = act_limit * mlp(x, list(hid_size) + [act_dim], activ, output_activ)
    with tf.variable_scope('q1'):
        # for the Q value network, since the output is a scalar, we don't need to scale
        # the scale it, so no output activation func is used
        q1 = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hid_size) + [1], activ, None), axis=1)
    with tf.variable_scope('q2'):
        q2 = tf.squeeze(mlp(tf.concat([x, a], axis=-1), list(hid_size) + [1], activ, None), axis=1)
    with tf.variable_scope('q1', reuse=True):
        q1_pi = tf.squeeze(mlp(tf.concat([x, pi], axis=-1), list(hid_size) + [1], activ, None), axis=1)
    return pi, q1, q2, q1_pi
