import os
import random
import numpy as np


def read_data(data_path, label_path,
              num_steps, num_sample):
  data = np.load(data_path)
  keys = data.keys()
  train_keys = keys[:9]
  test_keys = keys[9:]

  label = np.load(label_path)
  train_labels = label[:9]
  test_labels = label[9:]

  # 开始做数据
  train_data = []
  train_label = []
  for key, label in zip(train_keys, train_labels):
    cur_data = data[key]
    cur_data = np.transpose(cur_data, [1, 0, 2])
    # cur_data = cur_data[:num_steps, :, :]
    cur_data = np.reshape(cur_data, [-1, 310])

    for i in range(num_sample):
      idx = random.sample(range(cur_data.shape[0]), num_steps)
      idx.sort()

      sampled_data = cur_data[idx]
      train_data.append(list(sampled_data))
      train_label.append(label)
  train_data = np.array(train_data)

  test_data = []
  test_label = []
  for key, label in zip(test_keys, test_labels):
    cur_data = data[key]
    cur_data = np.transpose(cur_data, [1, 0, 2])
    # cur_data = cur_data[:num_steps, :, :]
    cur_data = np.reshape(cur_data, [-1, 310])

    for i in range(num_sample):
      idx = random.sample(range(cur_data.shape[0]), num_steps)
      idx.sort()

      sampled_data = cur_data[idx]
      test_data.append(list(sampled_data))
      test_label.append(label)
  test_data = np.array(test_data)

  return train_data, train_label, \
         test_data, test_label


if __name__ == '__main__':
  data_folder = 'hw4_data'
  data = np.load(os.path.join(data_folder, '01.npz'))

  train_data, train_label, \
  test_data, test_label = read_data(os.path.join(data_folder, '01.npz'),
                                    os.path.join(data_folder, 'label.npy'),
                                    100, 100)
  print(train_data.shape)


  # label=np.load(os.path.join(data_folder,'label.npy'))
  # print(label)
