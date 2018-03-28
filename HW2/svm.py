import os
import time
import scipy.io

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_data(data_folder, data_filename, label_filename):
  data = scipy.io.loadmat(os.path.join(data_folder, data_filename + ".mat", ))[data_filename]
  label = scipy.io.loadmat(os.path.join(data_folder, label_filename + ".mat", ))[label_filename]

  label = [l[0] + 1 for l in label]
  return data, label


def standard_svm():
  # data_folder = '../HW1/hw1_data'
  # train_data, train_label = get_data(data_folder, 'train_data', 'train_label')
  # # print(len(train_label))
  # test_data, test_label = get_data(data_folder, 'test_data', 'test_label')
  # # print(len(test_label))


  data_folder = 'hw2_data'
  train_data = np.load(os.path.join(data_folder, 'train_data.npy'))
  train_label = np.load(os.path.join(data_folder, 'train_label.npy'))

  test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
  test_label = np.load(os.path.join(data_folder, 'test_label.npy'))

  time0 = time.time()
  clf = SVC(kernel='rbf', C=1.0, probability=True)
  clf.fit(train_data, train_label)
  # print(clf.predict_proba(test_data))
  pred = clf.predict(test_data)
  print(pred)

  accuracy = accuracy_score(test_label, pred)  # 计算精确度
  print(accuracy)
  print(time.time() - time0)


def get_new_data(data, label, index_to_label):
  """
  one-vs-rest的数据划分方式
  :param train_data: 
  :param train_label: 
  :param n_classes: 
  :return: 
  """
  new_datas = [data for _ in range(len(index_to_label))]
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

  return new_datas, new_labels


def one_rest_svm():
  # data_folder = '../HW1/hw1_data'
  # train_data, train_label = get_data(data_folder, 'train_data', 'train_label')
  # test_data, test_label = get_data(data_folder, 'test_data', 'test_label')

  data_folder = 'hw2_data'
  train_data = np.load(os.path.join(data_folder, 'train_data.npy'))
  train_label = np.load(os.path.join(data_folder, 'train_label.npy'))

  test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
  test_label = np.load(os.path.join(data_folder, 'test_label.npy'))

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

  new_train_data, new_train_label = get_new_data(train_data, train_label, index_to_label)
  new_test_data, new_test_label = get_new_data(test_data, test_label, index_to_label)

  clfs = []
  for i in range(n_classes):
    clf = SVC(kernel='rbf', C=1.0, probability=True)
    clf.fit(new_train_data[i], new_train_label[i])
    clfs.append(clf)

  pred_probs = []
  for i in range(n_classes):
    pred_prob = clfs[i].predict_proba(new_test_data[i])  # 分别用n个clf预测n类测试数据，pred_probs.shape=[len(testdata),n,2]
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


if __name__ == '__main__':
  standard_svm()
  # one_rest_svm()
