import os
import random
import scipy.io
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


class MLPModel(object):
  def __init__(self, config):
    input_dim = config.input_dim

    self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='input')
    self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

    self.logits = self._get_logits(self.input, config)
    self.loss = self._build_loss(self.logits, self.label)
    self.train_op = self._build_train_op(self.loss, config)

    self.acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=self.label, k=1), tf.float32))

  def _get_logits(self, input, config):
    net = slim.fully_connected(input, config.hidden_size, activation_fn=tf.sigmoid, scope='hidden_layer')
    # net = slim.fully_connected(net, config.hidden_size, activation_fn=tf.tanh, scope='hidden_layer2')
    # net = slim.fully_connected(net, config.hidden_size, activation_fn=tf.sigmoid, scope='hidden_layer3')
    net = slim.fully_connected(net, config.output_dim, activation_fn=None, scope='output_layer')

    return net

  def _build_loss(self, logits, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss = tf.reduce_mean(loss)

    return loss

  def _build_train_op(self, loss, config):
    train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss)

    return train_op


def get_data(data_folder, data_filename, label_filename):
  data = scipy.io.loadmat(os.path.join(data_folder, data_filename + ".mat", ))[data_filename]
  label = scipy.io.loadmat(os.path.join(data_folder, label_filename + ".mat", ))[label_filename]

  # do_norm = True
  # if do_norm:
  #   new_data = []
  #   for d in data:
  #     max_d = max(d)
  #     min_d = min(d)
  #
  #     d = [(num - min_d) / (max_d - min_d) for num in d]
  #     new_data.append(d)
  #
  #   data = new_data
  label = [l[0] + 1 for l in label]
  return data, label


class MLPConfig(object):
  def __init__(self, batch_size, hidden_size, learnig_rate):
    self.batch_size = batch_size
    self.learning_rate = learnig_rate
    self.init_scale = 0.05

    self.hidden_size = hidden_size
    self.input_dim = 310
    self.output_dim = 3


def run_epoch(data, label, batch_size,
              sess, model):
  times = len(label) // batch_size
  temp_tuple = [(data[i], label[i]) for i in range(len(label))]
  random.shuffle(temp_tuple)

  data = [a for a, b in temp_tuple]
  label = [b for a, b in temp_tuple]

  loss_sum = 0
  acc_sum = 0
  for i in range(times):
    input_data = data[i * batch_size:(i + 1) * batch_size]
    target_label = label[i * batch_size:(i + 1) * batch_size]

    _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                            {model.input: input_data,
                             model.label: target_label})
    loss_sum += loss
    acc_sum += acc
  print('trainLoss %.3f  trainAcc %.3f' % (loss_sum / times, acc_sum / times))


def train():
  data_folder = 'hw1_data'
  train_data, train_label = get_data(data_folder, 'train_data', 'train_label')
  print(len(train_label))
  test_data, test_label = get_data(data_folder, 'test_data', 'test_label')
  print(len(test_label))

  hid_size = 100
  lr = 1e-2
  train_config = MLPConfig(64, hidden_size=hid_size, learnig_rate=lr)
  test_config = MLPConfig(len(test_label), hidden_size=hid_size, learnig_rate=lr)

  initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      train_model = MLPModel(config=train_config)

  with tf.name_scope('Valid'):
    with tf.variable_scope("Model", reuse=True):
      test_model = MLPModel(config=test_config)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100):
      run_epoch(train_data, train_label, train_config.batch_size,
                sess, train_model)

      test_loss, test_acc = sess.run([test_model.loss, test_model.acc],
                                     {test_model.input: test_data,
                                      test_model.label: test_label})
      print('testLoss %.3f  testAcc %.3f' % (test_loss, test_acc))
      print()
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  tf.set_random_seed(2)
  np.random.seed(2)
  train()
