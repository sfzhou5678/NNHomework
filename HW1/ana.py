import matplotlib.pyplot as plt

# 1. 横轴为hiddenSize, 纵轴为最大准确率/训练耗时
x_axix = range(10, 200, 10)
y_acc_dict = {}
y_time_dict = {}
for x in x_axix:
  y_acc_dict[x] = 0
for x in x_axix:
  y_time_dict[x] = 0

with open('res_file.log') as f:
  lines = f.readlines()
  for line in lines:
    hidden_size, learning_rate, acc, time = line.strip().split('\t')
    hidden_size = int(hidden_size)
    learning_rate = float(learning_rate)
    acc = float(acc)
    time = float(time)

    # if learning_rate not in y_acc_dict:
    #   y_acc_dict[learning_rate] = []
    # y_acc_dict[learning_rate].append(acc)
    #
    # if learning_rate not in y_time_dict:
    #   y_time_dict[learning_rate] = []
    # y_time_dict[learning_rate].append(time)
    y_acc_dict[hidden_size] += acc
    y_time_dict[hidden_size] += time

y_acc = [y_acc_dict[x] / 10 for x in x_axix]
y_time= [y_time_dict[x] / 10 for x in x_axix]
# sub_axix = filter(lambda x: x % 200 == 0, x_axix)
plt.title('Result Analysis')


# for key in y_acc_list:
plt.plot(x_axix, y_acc , color='red', label='acc')
plt.plot(x_axix, y_time, color='blue', label='time')
plt.grid()
# for key in y_time_dict:
#   plt.plot(x_axix, y_time_dict[key], color='green', label='')
plt.legend()  # 显示图例

plt.xlabel('hidden unit size')
plt.ylabel('')
plt.show()

# fig = plt.figure()
#
# ax1 = fig.add_subplot(111)
# ax1.plot(x_axix, y_acc,label='acc')
# ax1.legend()
# ax1.set_ylabel('Y values for exp(-x)')
#
# ax2 = ax1.twinx()  # this is the important function
# ax2.plot(x_axix, y_time, 'r',label='time')
# ax2.legend()
# # ax2.set_xlim([0, np.e])
# ax2.set_ylabel('Y values for ln(x)')
# ax2.set_xlabel('hidden Size')
# plt.grid()
# # plt.legend()
# plt.show()