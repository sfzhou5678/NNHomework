from sklearn.svm import SVC
import numpy as np


class MinModule(object):
  def __init__(self):
    self.clfs = []

  def clear(self):
    self.clfs = []

  def add_clf(self, train_data, train_label):
    clf = SVC(kernel='rbf', C=1.0, probability=True)
    clf.fit(train_data, train_label)
    self.clfs.append(clf)

  def pred(self, data):
    pred_probs = []
    for clf in self.clfs:
      pred_prob = clf.predict_proba(data)  # 分别用n个clf预测n类测试数据，pred_probs.shape=[len(testdata),n,2]
      pred_probs.append(pred_prob)

    total_pred_prob = []
    for i in range(len(data)):
      prob = [pred_probs[j][i][0] for j in range(len(self.clfs))]  # 共[n ,len(clfs)]个概率, 下面取每行的最小值，共产生n个概率
      total_pred_prob.append(prob)

    final_prob = np.min(total_pred_prob, axis=-1)
    return final_prob
