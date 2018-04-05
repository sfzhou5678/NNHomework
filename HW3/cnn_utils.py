import tensorflow as tf


def weight_variable(shape):
  return tf.get_variable('weight', shape,
                         initializer=tf.truncated_normal_initializer(stddev=0.1))


def bias_variable(shape):
  return tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.01))


def conv_2d(input, w_shape, b_shape, strides, name, padding='SAME', act_func=tf.nn.relu):
  with tf.variable_scope(name):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    conv = tf.nn.conv2d(input, w, strides=strides, padding=padding)
    conv = tf.nn.bias_add(conv, b)

    if act_func != None:
      conv = act_func(conv)

    return conv


def max_pool_2d(x, ksize, strides, name, padding='SAME'):
  with tf.variable_scope(name):
    return tf.nn.max_pool(x, ksize=ksize,
                          strides=strides, padding=padding)


def fully_connected(name, input, w_shape, b_shape, regularizer, need_dropout=False, keep_prob=1.0,
                    act_function=tf.nn.relu):
  with tf.variable_scope(name):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    if regularizer != None:
      tf.add_to_collection('losses', regularizer(w))

    fc = tf.matmul(input, w)
    fc += b

    if act_function != None:
      fc = act_function(fc)

    if need_dropout:
      fc = tf.nn.dropout(fc, keep_prob)
    return fc
