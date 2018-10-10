import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # dW为梯度的偏导
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # 类别数
  num_train = X.shape[0] # 训练集数目
  loss = 0.0
  for i in range(num_train): # 遍历训练集
    scores = X[i].dot(W) # 得分，1*10
    correct_class_score = scores[y[i]]
    for j in range(num_classes): # 遍历每个类别
      if j == y[i]: # 一举两得（对于损失函数和梯度）
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1，设置delta为1
      if margin > 0: # 当大于0时，指示函数为1，判断错误，需要对W矩阵进行变化
        loss += margin # 可以看出：梯度是一列一列更新的
        dW[:, j] += X[i].T # 看公式，对于单个i，j有很多个而y[i]仅一个
        dW[:, y[i]] -= X[i].T # j与y[i]不会相等，因为第33行

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W # 对正则化项的W求偏导，对一列中每个元素加一项

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # 首先是loss的计算
  num_train = X.shape[0]
  y_f = X.dot(W) # 得分矩阵
  # 正确类别的得分，同时(N,) - > (N, 1)
  y_c = y_f[range(num_train), list(y)].reshape(-1, 1)
  margins = np.maximum(y_f - y_c + 1, 0) # shape N*C，同时利用了广播
  margins[range(num_train), list(y)] = 0 # 根据公式，将这些元素置0
  loss = np.sum(margins) / num_train + reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = margins
  mask[margins > 0] = 1 # shape N*C
  # 由上面循环，当margin>0时，dW第j列要加X.T，第y[i]列要减X.T（行i为训练集遍历，列j为类别遍历）
  # 用上面的循环来理解下面的代码，同时画图理解，每次加或减的单位为X.T的一列
  # 将行的和的负数赋值给真实标记所在位置，每行每列只有一个（相当于把所有要减X.T的操作合并起来）
  # 除了y[i]列，其它列只有可能为1或0（因为根据公式，对于一个i，y[i]列会叠加X.T，而其它列仅一次）
  mask[range(num_train), list(y)] = -np.sum(mask, axis=1)
  # 点乘之后，对于X.T的每一列操作是一样的，和之前循环等价（深刻理解点乘）
  # 点乘：https://blog.csdn.net/alexxie1996/article/details/79184596的最后一张图
  dW = (X.T).dot(mask)
  dW = dW / num_train + 2 * reg * W
  # 考虑mask的一行i，对于其第j列，这个数字代表dW要加或减多少倍的X[i].T
  # X.T可以加或减到任何dW列
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
