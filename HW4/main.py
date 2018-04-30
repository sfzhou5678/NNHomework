import os
import random
import tensorflow as tf
import numpy as np

from data_reader import read_data


def lstm_cell(hidden_size, is_training, keep_prob):
  cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
  # if is_training and keep_prob < 1:
  if is_training and keep_prob < 1:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
  return cell


def build_decoder_cell_with_att(encoder_outputs, encoder_final_state,
                                batch_size, max_length,
                                rnn_layers, hidden_size, keep_prob, is_training):
  # 如果只有decoder用了MultiRNNCell而encoder用的是BasicCell那么就会报错(不一致就会报错)
  decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
    [lstm_cell(hidden_size, keep_prob, is_training) for _ in range(rnn_layers)])

  ## Create an attention mechanism
  # TODO 这里的memory_sequence_length表示source_sequence_length(输入，不是输出targets)中的非PAD的长度
  memory = encoder_outputs
  attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    hidden_size, memory,
    # memory_sequence_length=seq_lengths,
    memory_sequence_length=tf.constant(max_length, shape=[batch_size],
                                       dtype=tf.int32))

  decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=hidden_size,
    alignment_history=not is_training,
    # alignment_history=False,  # test时为true
    name="attention")

  attention_states = decoder_cell.zero_state(batch_size, tf.float32).clone(
    cell_state=encoder_final_state)
  decoder_init_state = attention_states

  return decoder_cell, decoder_init_state


class Config(object):
  def __init__(self, batch_size, hidden_size, num_steps):
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.num_steps = num_steps

    self.feature_dim = 310

    self.n_classes = 3

    self.rnn_layers = 2
    self.keep_prob = 1.0
    self.learning_rate = 1e-1
    self.lr_decay = 0.95
    self.max_grad_norm = 2
    self.init_scale = 0.05


class LSTM(object):
  def __init__(self, config, is_training):
    bi_lstm = False
    use_attention = False
    use_focal_loss = False
    gamma = 2

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
    # encoder_outputs = []
    # with tf.variable_scope("RNN") as encoder_scope:
    #
    #   for time_step in range(num_steps):
    #     if time_step > 0:
    #       tf.get_variable_scope().reuse_variables()
    #     (cell_output, state) = cell(self.input[:, time_step, :], state)
    #     encoder_outputs.append(cell_output)
    #   # 这里只拿了最后一个时刻的output来做分类
    #   final_output = cell_output
    #   encoder_final_state = state
    with tf.variable_scope("Encoder") as encoder_scope:
      if not bi_lstm:
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
          [lstm_cell(hidden_size, is_training, config.keep_prob) for _ in range(config.rnn_layers)])
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
          encoder_cell, self.input,
          sequence_length=tf.constant(num_steps, shape=[config.batch_size], dtype=tf.int32),
          dtype=tf.float32, scope=encoder_scope)
      else:
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
          [lstm_cell(hidden_size // 2, is_training, config.keep_prob) for _ in range(config.rnn_layers)])
        bw_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
          [lstm_cell(hidden_size // 2, is_training, config.keep_prob) for _ in range(config.rnn_layers)])

        encoder_outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
          encoder_cell, bw_encoder_cell,
          self.input,
          sequence_length=tf.constant(num_steps, shape=[config.batch_size], dtype=tf.int32),
          dtype=tf.float32, scope=encoder_scope)

        state = []
        for i in range(config.rnn_layers):
          fs = fw_state[i]
          bs = bw_state[i]
          encoder_final_state_c = tf.concat((fs.c, bs.c), 1)
          encoder_final_state_h = tf.concat((fs.h, bs.h), 1)
          encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h)
          state.append(encoder_final_state)
        encoder_final_state = tuple(state)

        if use_attention:
          encoder_outputs = tf.maximum(encoder_outputs[0], encoder_outputs[1])
        else:
          encoder_outputs = tf.concat([encoder_outputs[0], encoder_outputs[1]], axis=-1)

    final_output = encoder_outputs[:, -1, :]
    if use_attention:
      decoder_cell, decoder_init_state = build_decoder_cell_with_att(encoder_outputs, encoder_final_state,
                                                                     config.batch_size, config.num_steps,
                                                                     config.rnn_layers, config.hidden_size,
                                                                     config.keep_prob,
                                                                     is_training)
      GO_ID_embedding = tf.zeros([config.batch_size, config.feature_dim])
      state = decoder_init_state
      with tf.variable_scope("Decoder"):
        (cell_output, state) = decoder_cell(GO_ID_embedding, state)
      final_output = cell_output

    # pred
    softmax_w = tf.get_variable("softmax_w", [hidden_size, config.n_classes], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [config.n_classes], dtype=tf.float32)
    logits = tf.matmul(final_output, softmax_w) + softmax_b

    if use_focal_loss:
      ones = tf.one_hot(self.label, depth=config.n_classes)

      sigmoid_logits = tf.nn.softmax(logits)
      # - sigmoid_logits * ones
      loss = -((ones - sigmoid_logits * ones) ** gamma) * tf.log(tf.clip_by_value(sigmoid_logits, 1e-8, 1.0))
      self.loss = tf.reduce_sum(loss) / config.batch_size
    else:
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
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
  num_steps = 180

  data_folder = 'hw4_data'
  train_data, train_label, \
  test_data, test_label = read_data(os.path.join(data_folder, '01.npz'),
                                    os.path.join(data_folder, 'label.npy'),
                                    num_steps=num_steps, num_sample=100)

  # test_data=train_data
  # test_label=train_label

  train_config = Config(batch_size=32, hidden_size=100, num_steps=num_steps)
  test_config = Config(batch_size=len(test_label), hidden_size=100, num_steps=num_steps)

  initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      train_model = LSTM(train_config, is_training=True)

  # with tf.name_scope('Test'):
  #   with tf.variable_scope("Model", reuse=True, initializer=initializer):
  #     test_model = LSTM(test_config, is_training=False)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1000):
      print('epoch :', i + 1)
      run_epoch(train_data, train_label, train_config.batch_size,
                sess, train_model)
      # test_loss, test_acc = sess.run([test_model.loss, test_model.acc],
      #                                {test_model.input: test_data,
      #                                 test_model.label: test_label})
      # print('testLoss %.3f  testAcc %.3f' % (test_loss, test_acc))
      # print()

    coord.request_stop()
    coord.join(threads)
