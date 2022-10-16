from re import M
import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    num_images = 60000
    with gzip.open(image_filesname) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 28 * 28)
        data = (data / 255)
        
    with gzip.open(label_filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return data, labels
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR CODE
    m, n = Z.shape
    # This code raises error in backward Reshape()
    # logits = ndl.exp(Z)
    # softmax = logits / ndl.summation(logits, axes=(1,)).reshape(shape=(m, 1))
    # loss = -(y_one_hot * ndl.log(softmax)).sum() / ndl.Tensor([m])
    # return loss
    loss = ndl.log(ndl.exp(Z).sum(axes=(1, ))) - (Z * y_one_hot).sum(axes=(1,))
    # return loss.sum() / ndl.Tensor([m])
    return loss.sum() / m
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

   ### BEGIN YOUR CODE
    batches = np.array_split(X, len(X) // batch)
    labels = np.array_split(y, len(y) // batch)
    for batch_features, batch_labels in zip(batches, labels):
        m = batch_features.shape[0]
        num_classes = W2.shape[1]
        one_hot = np.zeros((batch_labels.size, num_classes))
        one_hot[np.arange(batch_labels.size), batch_labels] = 1

        tensor_batch = ndl.Tensor(batch_features)
        tensor_one_hot_labels = ndl.Tensor(one_hot)
        
        # forward pass
        Z = tensor_batch @ W1
        Z = ndl.relu(Z)
        Z = Z @ W2
        
        batch_loss = softmax_loss(Z, tensor_one_hot_labels)
        batch_loss.backward()
        
        W1_numpy = W1.numpy()
        W2_numpy = W2.numpy()
        
        W1_numpy -= lr * W1.grad.numpy()
        W2_numpy -= lr * W2.grad.numpy()
        
        W1 = ndl.Tensor(W1_numpy)
        W2 = ndl.Tensor(W2_numpy)
        
    return W1, W2
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
