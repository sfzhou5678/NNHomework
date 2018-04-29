import os
import numpy as np

if __name__ == '__main__':
  data_folder = 'hw4_data'

  data = np.load(os.path.join(data_folder, '03.npz'))
  keys = data.keys()
  print(data[keys[0]].shape)
  print(data)

  # labels=np.load(os.path.join(data_folder,'label.npy'))
  # print(labels.shape)
  # print(labels)
