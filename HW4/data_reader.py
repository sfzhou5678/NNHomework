import os
import numpy as np


def read_data(data_path, num_steps):
  data = np.load(data_path)
  keys = data.keys()
  train_keys = keys[:9]
  test_keys = keys[9:]

  train_data = []
  for key in train_keys:
    # TODO: pad
    cur_data = data[key]
    cur_data = np.transpose(cur_data, [1, 0, 2])
    cur_data = cur_data[:num_steps, :, :]
    cur_data = np.reshape(cur_data, [-1, 310])
    train_data.append(list(cur_data))
  train_data = np.array(train_data)

  test_data = []
  for key in test_keys:
    # TODO: pad
    cur_data = data[key]
    cur_data = np.transpose(cur_data, [1, 0, 2])
    cur_data = cur_data[:num_steps, :, :]
    cur_data = np.reshape(cur_data, [-1, 310])
    test_data.append(list(cur_data))
  test_data = np.array(test_data)

  return train_data, test_data


if __name__ == '__main__':
  data_folder = 'hw4_data'
  data = np.load(os.path.join(data_folder, '01.npz'))

  train_data, test_Data = read_data(os.path.join(data_folder, '01.npz'), 150)
  print(train_data.shape)


  # label=np.load(os.path.join(data_folder,'label.npy'))
  # print(label)
