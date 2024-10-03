import numpy as np
import pickle

def load_fmnist_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    dim = np.prod(X_train.shape[1:])
    X_train = X_train.reshape(-1, dim)
    X_val = X_val.reshape(-1, dim)
    X_test = X_test.reshape(-1, dim)

    return X_train, y_train, X_val, y_val, X_test, y_test

# For gradient check
def relative_error(x, y, h):
    h = h or 1e-12
    if type(x) is np.ndarray and type(y) is np.ndarray:
        top = np.abs(x - y)
        bottom = np.maximum(np.abs(x) + np.abs(y), h)
        return np.amax(top/bottom)
    else:
        return abs(x - y) / max(abs(x) + abs(y), h)


def numeric_gradient(f, x, df, eps):
    df = df or 1.0
    eps = eps or 1e-8
    n = x.size
    x_flat = x.reshape(n)
    dx_num = np.zeros(x.shape)
    dx_num_flat = dx_num.reshape(n)
    for i in range(n):
        orig = x_flat[i]
    
        x_flat[i] = orig + eps
        pos = f(x)
        if type(df) is np.ndarray:
            pos = pos.copy()
    
        x_flat[i] = orig - eps
        neg = f(x)
        if type(df) is np.ndarray:
            neg = neg.copy()

        d = (pos - neg) * df / (2 * eps)
        
        dx_num_flat[i] = d
        x_flat[i] = orig
    return dx_num

# Criterion for testing the modules
class TestCriterion(object):
    def __init__(self):
        return
        
    def forward(self, _input, _target):
        return np.mean(np.sum(np.abs(_input), 1))
    
    def backward(self, _input, _target):
        self._gradInput = np.sign(_input) / len(_input)
        return self._gradInput

# Vanilla SGD Optimizer
def sgd(x, dx, lr, weight_decay = 0):
    if type(x) is list:
        assert len(x) == len(dx), 'Should be the same'
        for _x, _dx in zip(x, dx):
            sgd(_x, _dx, lr)
    else:
        x -= lr * (dx + 2 * weight_decay * x)  



# Utilities for training, prediction, and measuring accuracy
        
def predict(X, model):
    """
    Evaluate the soft predictions of the model.
    Input:
    X : N x d array (no unit terms)
    model : a multi-layer perceptron
    Output:
    yhat : N x C array
        yhat[n][:] contains the score over C classes for X[n][:]
    """
    return model.forward(X)

def error_rate(X, Y, model):
    """
    Compute error rate (between 0 and 1) for the model
    Y needs to be one-hot labels for this to work
    """
    res = 1 - (model.forward(X).argmax(-1) == Y.argmax(-1)).mean()
    return res

from copy import deepcopy

def train(model, criterion, X_train, y_train, X_val, y_val, trainopt):
    """
    Run the train + evaluation on a given train/val partition
    trainopt: various (hyper)parameters of the training procedure
    During training, choose the model with the lowest validation error. (early stopping)
    """
    
    lr = trainopt['lr']
    
    N = X_train.shape[0] # number of data points in X
    
    # Save the model with lowest validation error
    min_val_error = np.inf
    saved_model = None
    
    shuffled_idx = np.random.permutation(N)
    start_idx = 0
    for iteration in range(trainopt['maxiter']):
        if iteration % int(trainopt['lr_decay_interval'] * trainopt['maxiter']) == 0:
            lr *= trainopt['lr_decay']

        # form the next mini-batch
        stop_idx = min(start_idx + trainopt['batch_size'], N)
        batch_idx = range(N)[int(start_idx):int(stop_idx)]
        xs = X_train[shuffled_idx[batch_idx],:]
        ys = y_train[shuffled_idx[batch_idx],:]

        outputs = model.forward(xs)
        loss = criterion.forward(outputs, ys)
        # print(loss)
        d_outputs = criterion.backward(outputs, ys)
        model.backward(xs, d_outputs)
        
        # Update the data using 
        params, d_params = model.parameters()
        sgd(params, d_params, lr, weight_decay = trainopt['weight_decay'])
        start_idx = stop_idx % N
        
        if (iteration % trainopt['display_iter']) == 0:
            # compute train and val error; multiply by 100 for readability (make it percentage points)
            train_error = 100 * error_rate(X_train, y_train, model)
            val_error = 100 * error_rate(X_val, y_val, model)
            print('{:8} batch loss: {:.3f} train error: {:.3f} val error: {:.3f}'.format(iteration, loss, train_error, val_error))
            
            if val_error < min_val_error:
                saved_model = deepcopy(model)
                min_val_error = val_error
        
    return saved_model, min_val_error, train_error
