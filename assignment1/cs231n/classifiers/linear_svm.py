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
  dWn = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    dW = []
    dWy = 0.0
    for j in xrange(num_classes):
      if j == y[i]:        
        continue
      else:
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        dWy += (margin>0)
        dwjx = (margin>0)*X[i]
        dW.extend(dwjx.reshape([1,W.shape[0]]))
      
      if margin > 0:
        loss += margin
      else:
        continue
        
    dwyx = -dWy*X[i]
    dW.insert(y[i],dwyx)
    dW1 = np.array(dW)
    dWn += dW1.T
  dWn /= num_train
  dW = dWn
   
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  XW = X.dot(W)
    
  abc = [XW[i,:] - XW[i,y[i]]+1 for i in range(num_train)]
  abc = np.array(abc)    
  abc1 = np.maximum(0,abc)
  for i in range(num_train):
    abc1[i,y[i]] = 0
    
  loss = np.sum(abc1)/num_train

  loss += 0.5 * reg * np.sum(W * W)
  
  
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
  
  for i in range(num_train):
    abc[i,y[i]] = 0
    
  abc2 = abc>0
  abc2 = abc2.astype('float')
  
  for i in range(num_train):
    abc2[i,y[i]] = - abc2[i,:].sum()
    
  dW = 0
  for i in range(num_train):
        b = X[i,:].repeat(num_classes).reshape([-1,num_classes])
        dW += b*abc2[i,:]
  dW /= num_train


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
