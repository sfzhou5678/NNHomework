import os
import numpy as np

data_folder = 'HW2/hw2_data'
test_data = np.load(os.path.join(data_folder, 'test_data.npy'))
print(test_data)
