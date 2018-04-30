import os
import random
import tensorflow as tf
import numpy as np

from data_reader import read_data


def lstm_cell(hidden_size, is_training, keep_prob):
  cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
  if is_training and keep_prob < 1:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
  return cell


class Config(object):
  def __init__(self, batch_size, hidden_size, num_steps):
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.num_steps = num_steps

    self.feature_dim = 310

    self.n_classes = 3

    self.rnn_layers = 2
    self.keep_prob = 0.5
    self.learning_rate = 1e-2
    self.lr_decay = 0.95
    self.max_grad_norm = 5
    self.init_scale = 0.05


class LSTM(object):
  def __init__(self, config, is_training):
    hidden_size = config.hidden_size
    num_steps = config.num_steps
    feature_dim = config.feature_dim
    self.input = tf.placeholder(tf.float32, [None, num_steps, feature_dim], 'input')
    self.label = tf.placeholder(tf.int32, [None], 'label')

    # cell
    cell = tf.nn.rnn_cell.MultiRNNCell(
      [lstm_cell(hidden_size, is_training, config.keep_prob) for _ in range(config.rnn_layers)])
    init_state = cell.zero_state(config.batch_size, tf.float32)
    state = init_state

    # for
    # TODO: BiLSTM
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(self.input[:, time_step, :], state)
        if time_step == num_steps - 1:
          # 这里只拿了最后一个时刻的output来做分类
          output = cell_output

    # pred
    softmax_w = tf.get_variable("softmax_w", [hidden_size, config.n_classes], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [config.n_classes], dtype=tf.float32)
    logits = tf.matmul(output, softmax_w) + softmax_b

    # loss + train op
    self.loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label))

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
      config.learning_rate,
      global_step,
      50,
      config.lr_decay
    )

    # 控制梯度大小，定义优化方法和训练步骤。
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    # acc
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), dtype=tf.int32), self.label)
    self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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


if __name__ == '__main__':
  num_steps = 150

  data_folder = 'hw4_data'
  train_data, test_data = read_data(os.path.join(data_folder, '01.npz'), num_steps)

  # FIXME: 临时的label读取，应该和data合到一块
  label = np.load(os.path.join(data_folder, 'label.npy'))
  train_label = label[:9]
  test_label = label[9:]

  train_config = Config(batch_size=2, hidden_size=100, num_steps=num_steps)
  test_config = Config(batch_size=len(test_label), hidden_size=100, num_steps=num_steps)

  initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      train_model = LSTM(train_config, is_training=True)

  with tf.name_scope('Test'):
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      test_model = LSTM(test_config, is_training=False)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1000):
      run_epoch(train_data, train_label, train_config.batch_size,
                sess, train_model)
      test_loss, test_acc = sess.run([test_model.loss, test_model.acc],
                                     {test_model.input: test_data,
                                      test_model.label: test_label})
      print('testLoss %.3f  testAcc %.3f' % (test_loss, test_acc))
      print()

    coord.request_stop()
    coord.join(threads)
