import os
import time
import random

import tensorflow as tf
import numpy as np

from data_reader import *
from cnn_utils import *

slim = tf.contrib.slim


class MnistModel(object):
  def __init__(self, config):
    input_dim = config.input_dim

    if config.model_type == 'MLP':
      self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='input')
      self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

      self.logits = self._get_mlp_logits(self.input, config)
    elif config.model_type == 'LeNet':
      self.input = tf.placeholder(dtype=tf.float32, shape=[None, config.width, config.height, config.input_dim],
                                  name='input')
      self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

      self.logits = self._get_lenet_logits(self.input, config)

    self.loss = self._build_loss(self.logits, self.label)
    self.train_op = self._build_train_op(self.loss, config)

    self.acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=self.label, k=1), tf.float32))

  def _get_mlp_logits(self, input, config):
    net = slim.fully_connected(input, config.hidden_size, activation_fn=tf.sigmoid, scope='hidden_layer')
    net = slim.fully_connected(net, config.output_dim, activation_fn=None, scope='output_layer')

    return net

  def _get_lenet_logits(self, input, config):
    DEPTH1 = 16
    DEPTH2 = DEPTH1 * 2

    network = slim.conv2d(input, DEPTH1, 5)
    network = slim.conv2d(network, DEPTH2, 5)

    network = slim.flatten(network)
    network = slim.fully_connected(network, 120, activation_fn=tf.nn.relu)
    network = slim.fully_connected(network, 84, activation_fn=tf.nn.relu)
    network = slim.fully_connected(network, config.output_dim, activation_fn=tf.nn.relu)

    # # 以前的自制conv工具模块(效果比slim略差1%(97.5%), 但是速度快一倍(75s))
    # network = conv_2d(input, [5, 5, config.input_dim, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv', padding='VALID')
    # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer2-pool')
    # network = conv_2d(network, [5, 5, DEPTH1, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer3-conv', padding='VALID')
    # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer4-pool')
    #
    # pool_shape = network.get_shape().as_list()
    # nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # network = tf.reshape(network, [-1, nodes])
    # network = fully_connected('layer5-fc', network, [nodes, 120], [120], regularizer=None, need_dropout=False)
    # network = fully_connected('layer6-fc', network, [120, 84], [84], regularizer=None, need_dropout=False)
    # network = fully_connected('layer7-fc', network, [84, config.output_dim], [config.output_dim], regularizer=None,
    #                           need_dropout=False, act_function=None)

    logits = network
    return logits

  def _build_loss(self, logits, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss = tf.reduce_mean(loss)

    return loss

  def _build_train_op(self, loss, config):
    train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss)

    return train_op


class MLPConfig(object):
  def __init__(self, batch_size, hidden_size, learnig_rate, model_type):
    self.model_type = model_type

    self.batch_size = batch_size
    self.learning_rate = learnig_rate
    self.init_scale = 0.05

    self.hidden_size = hidden_size
    self.input_dim = 28 * 28
    self.output_dim = 10


class LeNetConfig(MLPConfig):
  def __init__(self, batch_size, hidden_size, learnig_rate, model_type):
    super().__init__(batch_size, hidden_size, learnig_rate, model_type)

    self.width = 28
    self.height = 28
    self.input_dim = 1


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


def train_MLP():
  data_folder = 'mnist_data'
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  train_data = extract_images(os.path.join(data_folder, TRAIN_IMAGES))
  train_label = extract_labels(os.path.join(data_folder, TRAIN_LABELS))

  test_data = extract_images(os.path.join(data_folder, TEST_IMAGES))
  test_label = extract_labels(os.path.join(data_folder, TEST_LABELS))

  # 针对MLP, 将数据展平
  train_data = np.reshape(train_data, newshape=[-1, 28 * 28])
  test_data = np.reshape(test_data, newshape=[-1, 28 * 28])

  hidden_size = 200
  lr = 1e-3
  lr_no = 1

  train_config = MLPConfig(64, hidden_size=hidden_size, learnig_rate=lr, model_type='MLP')
  test_config = MLPConfig(len(test_label), hidden_size=hidden_size, learnig_rate=lr, model_type='MLP')

  initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model-%d-%d" % (hidden_size, lr_no), reuse=None, initializer=initializer):
      train_model = MnistModel(config=train_config)

  with tf.name_scope('Valid'):
    with tf.variable_scope("Model-%d-%d" % (hidden_size, lr_no), reuse=True):
      test_model = MnistModel(config=test_config)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    time0 = time.time()
    best_test_acc = 0
    for i in range(50):
      run_epoch(train_data, train_label, train_config.batch_size,
                sess, train_model)

      test_loss, test_acc = sess.run([test_model.loss, test_model.acc],
                                     {test_model.input: test_data,
                                      test_model.label: test_label})
      print('testLoss %.3f  testAcc %.3f' % (test_loss, test_acc))
      print()
      best_test_acc = max(best_test_acc, test_acc)

    # hidden_size, learning_rate, best_test_acc
    used_time = time.time() - time0
    print('{}\t{}\t{}\t{}'.format(train_config.hidden_size, train_config.learning_rate, best_test_acc, used_time))
    # res_file.write(
    #   '{}\t{}\t{}\t{}\n'.format(train_config.hidden_size, train_config.learning_rate, best_test_acc, used_time))
    # res_file.flush()

    coord.request_stop()
    coord.join(threads)


def trainLeNet():
  data_folder = 'mnist_data'
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  train_data = extract_images(os.path.join(data_folder, TRAIN_IMAGES))
  train_label = extract_labels(os.path.join(data_folder, TRAIN_LABELS))

  test_data = extract_images(os.path.join(data_folder, TEST_IMAGES))
  test_label = extract_labels(os.path.join(data_folder, TEST_LABELS))

  hidden_size = 200
  lr = 1e-3
  lr_no = 1

  train_config = LeNetConfig(64, hidden_size=hidden_size, learnig_rate=lr, model_type='LeNet')
  test_config = LeNetConfig(len(test_label), hidden_size=hidden_size, learnig_rate=lr, model_type='LeNet')

  initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model-%d-%d" % (hidden_size, lr_no), reuse=None, initializer=initializer):
      train_model = MnistModel(config=train_config)

  with tf.name_scope('Valid'):
    with tf.variable_scope("Model-%d-%d" % (hidden_size, lr_no), reuse=True):
      test_model = MnistModel(config=test_config)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    time0 = time.time()
    best_test_acc = 0
    for i in range(20):
      run_epoch(train_data, train_label, train_config.batch_size,
                sess, train_model)

      test_loss, test_acc = sess.run([test_model.loss, test_model.acc],
                                     {test_model.input: test_data,
                                      test_model.label: test_label})
      print('testLoss %.3f  testAcc %.3f' % (test_loss, test_acc))
      print()
      best_test_acc = max(best_test_acc, test_acc)

    # hidden_size, learning_rate, best_test_acc
    used_time = time.time() - time0
    print('{}\t{}\t{}\t{}'.format(train_config.hidden_size, train_config.learning_rate, best_test_acc, used_time))
    # res_file.write(
    #   '{}\t{}\t{}\t{}\n'.format(train_config.hidden_size, train_config.learning_rate, best_test_acc, used_time))
    # res_file.flush()

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  # train_MLP()
  trainLeNet()
