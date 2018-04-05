import os
import time
import scipy.io

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from MinMaxModule import MinModule


def get_data(data_folder, data_filename, label_filename):
  data = scipy.io.loadmat(os.path.join(data_folder, data_filename + ".mat", ))[data_filename]
  label = scipy.io.loadmat(os.path.join(data_folder, label_filename + ".mat", ))[label_filename]

  label = [l[0] + 1 for l in label]
  return data, label


def standard_svm(data_source):
  if data_source == 'hw1':
    data_folder = '../HW1/hw1_data'
    train_data, train_label = get_data(data_folder, 'train_data', 'train_label')
    test_data, test_label = get_data(data_folder, 'test_data', 'test_label')
  elif data_source == 'hw2':
    data_folder = 'hw2_data'
    train_data = np.load(os.path.join(data_folder, 'train_data.npy'))
    train_label = np.load(os.path.join(data_folder, 'train_label.npy'))

    test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
    test_label = np.load(os.path.join(data_folder, 'test_label.npy'))
  else:
    raise Exception('data source error')

  time0 = time.time()
  clf = SVC(kernel='rbf', C=1.0, probability=True)
  clf.decision_function_shape = 'ovr'
  clf.fit(train_data, train_label)
  # print(clf.predict_proba(test_data))
  pred = clf.predict(test_data)
  print(pred)

  accuracy = accuracy_score(test_label, pred)  # 计算精确度
  print(accuracy)
  print(time.time() - time0)


def get_new_labels(label, index_to_label):
  """
  one-vs-rest的数据划分方式
  :param train_data: 
  :param train_label: 
  :param n_classes: 
  :return: 
  """
  new_labels = []
  for i in range(len(index_to_label)):
    target_lbl = index_to_label[i]
    labels = []
    for lbl in label:
      if lbl == target_lbl:
        labels.append(0)
      else:
        labels.append(1)
    new_labels.append(labels)

  return new_labels


def one_rest_svm(data_source):
  if data_source == 'hw1':
    data_folder = '../HW1/hw1_data'
    train_data, train_label = get_data(data_folder, 'train_data', 'train_label')
    test_data, test_label = get_data(data_folder, 'test_data', 'test_label')
  elif data_source == 'hw2':

    data_folder = 'hw2_data'
    train_data = np.load(os.path.join(data_folder, 'train_data.npy'))
    train_label = np.load(os.path.join(data_folder, 'train_label.npy'))

    test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
    test_label = np.load(os.path.join(data_folder, 'test_label.npy'))
  else:
    raise Exception('data source error')

  time0 = time.time()
  n_classes = 3
  label_to_index = {}
  index_to_label = {}
  for l, d in zip(train_label, train_data):
    if l not in label_to_index:
      label_to_index[l] = len(label_to_index)
      index_to_label[len(label_to_index) - 1] = l
  print(label_to_index)
  print(index_to_label)
  assert n_classes == len(label_to_index)

  new_train_label = get_new_labels(train_label, index_to_label)

  clfs = []
  for i in range(n_classes):
    clf = SVC(kernel='rbf', C=1.0, probability=True)
    clf.fit(train_data, new_train_label[i])
    clfs.append(clf)

  pred_probs = []
  for i in range(n_classes):
    pred_prob = clfs[i].predict_proba(test_data)  # 分别用n个clf预测n类测试数据，pred_probs.shape=[len(testdata),n,2]
    pred_probs.append(pred_prob)

  total_pred_prob = []
  for i in range(len(test_label)):
    prob = [pred_probs[j][i][0] for j in range(n_classes)]
    total_pred_prob.append(prob)

  pred_index = np.argmax(total_pred_prob, axis=-1)
  print(pred_index)
  pred_label = [index_to_label[idx] for idx in pred_index]
  print(pred_label)

  accuracy = accuracy_score(test_label, pred_label)  # 最后和原始的test_label对比计算精确度
  print(accuracy)
  print(time.time() - time0)


def get_m3_data(train_data, train_label):
  labei_i_data = [[], []]

  for data, label in zip(train_data, train_label):
    if label == 0:
      labei_i_data[0].append(data)
    else:
      labei_i_data[1].append(data)
  min_size = min(len(labei_i_data[0]), len(labei_i_data[1]))

  label0_data = []
  n0 = max(1, round(len(labei_i_data[0]) / min_size))  # 数据0分成n块
  part_size0 = len(labei_i_data[0]) // n0
  for i in range(n0):
    label0_data.append(labei_i_data[0][i * part_size0:(i + 1) * part_size0])

  label1_data = []
  n1 = max(1, round(len(labei_i_data[1]) / min_size))  # 数据1分成n块
  part_size1 = len(labei_i_data[1]) // n1
  for i in range(n1):
    label1_data.append(labei_i_data[1][i * part_size1:(i + 1) * part_size1])

  new_train_data = []
  new_train_label = []
  for i in range(len(label0_data)):
    for j in range(len(label1_data)):
      data = []
      data += label0_data[i]
      data += label1_data[j]

      label = []
      for _ in range(len(label0_data[i])):
        label.append(0)
      for _ in range(len(label1_data[j])):
        label.append(1)

      new_train_data.append(data)
      new_train_label.append(label)

  return new_train_data, new_train_label
  # return [train_data[:len(train_data) // 2], train_data[len(train_data) // 2:]], \
  #        [train_label[:len(train_data) // 2], train_label[len(train_data) // 2:]]
  # return [train_data],[train_label]


def m3_svm(data_source):
  if data_source == 'hw1':
    data_folder = '../HW1/hw1_data'
    train_data, train_label = get_data(data_folder, 'train_data', 'train_label')
    test_data, test_label = get_data(data_folder, 'test_data', 'test_label')
  elif data_source == 'hw2':

    data_folder = 'hw2_data'
    train_data = np.load(os.path.join(data_folder, 'train_data.npy'))
    train_label = np.load(os.path.join(data_folder, 'train_label.npy'))

    test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
    test_label = np.load(os.path.join(data_folder, 'test_label.npy'))
  else:
    raise Exception('data source error')

  time0 = time.time()
  n_classes = 3
  label_to_index = {}
  index_to_label = {}
  for l, d in zip(train_label, train_data):
    if l not in label_to_index:
      label_to_index[l] = len(label_to_index)
      index_to_label[len(label_to_index) - 1] = l
  print(label_to_index)
  print(index_to_label)
  assert n_classes == len(label_to_index)

  # newLabel对应OvR模式下的K个二分类问题的标签
  new_train_label = get_new_labels(train_label, index_to_label)

  # region #... 最小模块
  # 下面需要将每个二分类问题拆分成多个类别平衡的二分类问题，最后取min作为预测的结果
  min_modules = []
  for i in range(n_classes):
    m3_train_data_list, m3_train_label_list = get_m3_data(train_data, new_train_label[i])
    min_module = MinModule()
    for j in range(len(m3_train_data_list)):
      min_module.add_clf(m3_train_data_list[j], m3_train_label_list[j])
    min_modules.append(min_module)

  # 这里会产生K个MinModules各自的经过min之后得到的final_prob,
  # pred_probs.shape=[K,n]
  pred_probs = []
  for min_module in min_modules:
    pred_prob = min_module.pred(test_data)
    pred_probs.append(pred_prob)
  # endregion

  # region #... 最大模块
  # 给K个最小模块取最大值，输入: [K,n] 输出[N]

  total_pred_prob = []
  for i in range(len(test_data)):
    prob = []
    for j in range(n_classes):
      prob.append(pred_probs[j][i])
    total_pred_prob.append(prob)
  pred_index = np.argmax(total_pred_prob, axis=-1)
  print(pred_index)
  pred_label = [index_to_label[idx] for idx in pred_index]
  print(pred_label)
  # endregion

  accuracy = accuracy_score(test_label, pred_label)  # 最后和原始的test_label对比计算精确度
  print(accuracy)
  print(time.time() - time0)


if __name__ == '__main__':
  standard_svm(data_source='hw2')
  # one_rest_svm(data_source='hw1')
  # m3_svm(data_source='hw2')
