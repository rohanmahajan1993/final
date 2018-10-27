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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_losses = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        num_losses +=1
    dW[:, y[i]] += -1 * num_losses * X[i]
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  dW /= num_train       
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W    

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  # compute the loss and the gradient
  num_train = X.shape[0]
  example_class = np.dot(X, W)
  dW = np.zeros(W.shape)
  real_class_value = np.array([example_class[index, y_value] for index, y_value in enumerate(y)]).reshape(num_train, 1)
  print (example_class.shape, real_class_value.shape)
  differences = example_class - real_class_value +1
  are_positive = differences > 0
  positive_numbers = differences * are_positive
  loss = np.sum(positive_numbers)
  loss -= num_train  ##the class is in included i.e. we don't handle i!=j so we must account for it now; each of these example will have loss of 1
  loss /= num_train
  loss += reg * np.sum(W * W)  
  dW /= num_train  
  dW += 2 * reg * W
  return loss, dW
