import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train): # 遍历训练集
    scores = X[i].dot(W)
    # 将所有得分减去最大值，于是所有分数小于0，防止数值过大
    adjust_scores = scores - np.max(scores)
    loss += -np.log(np.exp(adjust_scores[y[i]]) / np.sum(np.exp(adjust_scores)))
    for j in range(num_classes):
      prob = np.exp(adjust_scores[j]) / np.sum(np.exp(adjust_scores))
      if j == y[i]:
        dW[:, j] += (-1 + prob) * X[i]
      else:
        dW[:, j] += prob * X[i]
        
  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  # 得分减去每行最大值归一化
  adjust_scores = np.exp(scores - np.max(scores, axis=1).reshape(-1, 1))
  # 每行加起来得到每个数据的得分总和
  sum_scores = np.sum(adjust_scores, axis=1).reshape(-1, 1)
  class_prob = adjust_scores / sum_scores # 广播，shape N*C，得到的是概率
  prob = class_prob[range(num_train), list(y)] # 只取正确分类的损失，shape N*1
  total_loss = -np.log(prob) # shape N*1
  loss = np.sum(total_loss) / num_train + reg * np.sum(W * W)
  
  class_prob[range(num_train), list(y)] -= 1 # 求梯度时正确分类还需要再减X[i].T
  dW = (X.T).dot(class_prob) # 和SVM类似
  dW = dW / num_train + 2 * reg * W
  # 理解：X.T shape D*N，class_prob shape N*C
  # 考虑第一个训练数据，假设其正确分类为5，假设class_prob[1][5]=-1，在X.T中为第一列
  # 第一列的每个数字（假设第i个）都会乘到-1，乘完之后会处于dW的第i行第5列
  # 相当于对dW的一整列都减去了X[0].T
  # 再看公式：对于每个数据i（X.T中第i列），对于每个类别j（class_prob中第j列），对wj整列进行加或减X[i].T的倍数，和上面描述的一致
  # 以此可以类推其它数据在这个分类的操作
  # class_prob的第i行第j列就代表第i个训练数据在第j个类别的处理
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

