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
    pre = -0.5*(((x - mu) / (tf.exp(log_std) + EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre, axis=1)

def clip_but_pass_gradient(x, low=-1, high=1.):
    # this implementation achieves the similar functions as tf.clip_by_value
    clip_up = tf.cast(x > high, tf.float32)
    clip_low = tf.case(x < low, tf.float32)
    return x + tf.stop_gradient((high - x)* clip_up + (low - x)*clip_low)

""""
Policies
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20