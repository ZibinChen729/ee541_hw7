#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import h5py
import matplotlib.pyplot as plt



def load_hdf5_data(train_path, test_path):
    with h5py.File(train_path, 'r') as f:
        print("Train file keys:", list(f.keys()))
        X_all = f['xdata'][:]
        y_all = f['ydata'][:]

    with h5py.File(test_path, 'r') as f:
        print("Test file keys:", list(f.keys()))
        X_test = f['xdata'][:]
        y_test = f['ydata'][:]

    print("Raw X_all shape:", X_all.shape)
    print("Raw y_all shape:", y_all.shape)
    print("Raw X_test shape:", X_test.shape)
    print("Raw y_test shape:", y_test.shape)

    return X_all, y_all, X_test, y_test


def preprocess_x(X):
    X = np.array(X, dtype=np.float32)

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 2:
        pass
    else:
        raise ValueError("Unexpected X shape: {}".format(X.shape))

    if X.max() > 1.0:
        X = X / 255.0

    return X


def preprocess_y(y):
    y = np.array(y)


    if y.ndim == 2 and y.shape[1] == 10:
        y = np.argmax(y, axis=1)
    else:
        y = y.astype(np.int64).reshape(-1)

    return y.astype(np.int64)


def train_val_split(X, y, train_size=50000):

    X = X[:60000]
    y = y[:60000]

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:60000]
    y_val = y[train_size:60000]

    return X_train, y_train, X_val, y_val


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out



def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    grad[x <= 0] = 0.0
    return grad


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    t = np.tanh(x)
    return 1.0 - t * t


def get_activation(name):
    if name == 'relu':
        return relu, relu_derivative
    elif name == 'tanh':
        return tanh, tanh_derivative
    else:
        raise ValueError("Unsupported activation: {}".format(name))



def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def cross_entropy_loss(probs, y_true_onehot, eps=1e-12):
    probs = np.clip(probs, eps, 1.0)
    loss = -np.sum(y_true_onehot * np.log(probs)) / y_true_onehot.shape[0]
    return loss


def compute_accuracy(probs, y_true):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)


def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)


def he_init(in_dim, out_dim):
    std = np.sqrt(2.0 / in_dim)
    return (np.random.randn(in_dim, out_dim) * std).astype(np.float32)



class MLP:
    def __init__(self, layer_sizes, activation='relu', weight_decay=0.0, seed=42):
        """
        例如:
        layer_sizes = [784, 256, 128, 10]
        """
        np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation_name = activation
        self.act, self.act_deriv = get_activation(activation)
        self.weight_decay = weight_decay

        self.params = {}

        for i in range(1, len(layer_sizes)):
            in_dim = layer_sizes[i - 1]
            out_dim = layer_sizes[i]

            if i < len(layer_sizes) - 1:
                if activation == 'relu':
                    W = he_init(in_dim, out_dim)
                else:
                    W = xavier_init(in_dim, out_dim)
            else:
                W = xavier_init(in_dim, out_dim)

            b = np.zeros((1, out_dim), dtype=np.float32)

            self.params['W' + str(i)] = W
            self.params['b' + str(i)] = b

    def forward(self, X):
        cache = {}
        cache['A0'] = X
        A = X


        for i in range(1, self.num_layers):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]

            Z = A @ W + b
            A = self.act(Z)

            cache['Z' + str(i)] = Z
            cache['A' + str(i)] = A


        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]

        ZL = A @ W + b
        probs = softmax(ZL)

        cache['Z' + str(self.num_layers)] = ZL
        cache['A' + str(self.num_layers)] = probs

        return cache, probs

    def backward(self, cache, y_onehot):
        grads = {}
        m = y_onehot.shape[0]
        L = self.num_layers

        probs = cache['A' + str(L)]
        dZ = (probs - y_onehot) / m


        A_prev = cache['A' + str(L - 1)] if L - 1 >= 1 else cache['A0']
        grads['dW' + str(L)] = A_prev.T @ dZ + self.weight_decay * self.params['W' + str(L)]
        grads['db' + str(L)] = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ self.params['W' + str(L)].T


        for i in range(L - 1, 0, -1):
            Z = cache['Z' + str(i)]
            dZ = dA_prev * self.act_deriv(Z)

            A_prev = cache['A' + str(i - 1)] if i - 1 >= 1 else cache['A0']

            grads['dW' + str(i)] = A_prev.T @ dZ + self.weight_decay * self.params['W' + str(i)]
            grads['db' + str(i)] = np.sum(dZ, axis=0, keepdims=True)

            if i > 1:
                dA_prev = dZ @ self.params['W' + str(i)].T

        return grads

    def update_params(self, grads, lr):
        for i in range(1, self.num_layers + 1):
            self.params['W' + str(i)] -= lr * grads['dW' + str(i)]
            self.params['b' + str(i)] -= lr * grads['db' + str(i)]

    def predict_proba(self, X):
        _, probs = self.forward(X)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        _, probs = self.forward(X)
        y_onehot = one_hot(y, num_classes=10)

        loss = cross_entropy_loss(probs, y_onehot)
        acc = compute_accuracy(probs, y)

        return loss, acc



def get_minibatches(X, y, batch_size, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=100, initial_lr=0.01,
                decay_epochs=(20, 40), verbose=True):

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    lr = initial_lr

    for epoch in range(epochs):
        if epoch in decay_epochs:
            lr = lr / 2.0


        for Xb, yb in get_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True):
            yb_onehot = one_hot(yb, num_classes=10)

            cache, probs = model.forward(Xb)
            grads = model.backward(cache, yb_onehot)
            model.update_params(grads, lr)

        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)

        if verbose:
            print("Epoch {:02d} | lr={:.5f} | train_loss={:.4f}, train_acc={:.4f} | val_loss={:.4f}, val_acc={:.4f}".format(
                epoch + 1, lr, train_loss, train_acc, val_loss, val_acc
            ))

    return history



def plot_histories(results, decay_epochs=(20, 40)):
    for name, hist in results.items():
        plt.figure(figsize=(8, 5))

        epochs = np.arange(1, len(hist['train_acc']) + 1)

        plt.plot(epochs, hist['train_acc'], label='Training Accuracy')
        plt.plot(epochs, hist['val_acc'], label='Validation Accuracy')

        for e in decay_epochs:
            plt.axvline(x=e, linestyle='--', label='LR Decay' if e == decay_epochs[0] else None)

        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.title('Configuration: ' + name)
        plt.legend()
        plt.grid(True)
        plt.show()



def run_experiments(train_path, test_path):
    X_all, y_all, X_test, y_test = load_hdf5_data(train_path, test_path)

    X_all = preprocess_x(X_all)
    y_all = preprocess_y(y_all)
    X_test = preprocess_x(X_test)
    y_test = preprocess_y(y_test)

    X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, train_size=50000)

    print("Processed X_all:", X_all.shape)
    print("Processed y_all:", y_all.shape)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_val:", np.unique(y_val))
    print("Unique labels in y_test:", np.unique(y_test))

    layer_sizes = [784, 256, 128, 10]
    batch_size = 100
    epochs = 50
    decay_epochs = (20, 40)
    weight_decay = 1e-4

    activations = ['relu', 'tanh']
    learning_rates = [0.1, 0.01, 0.001]

    all_results = {}
    best_config = None
    best_val_acc = -1.0

    for act_name in activations:
        for lr in learning_rates:
            config_name = act_name + "_lr" + str(lr)

            print("\n" + "=" * 70)
            print("Training config:", config_name)
            print("=" * 70)

            model = MLP(
                layer_sizes=layer_sizes,
                activation=act_name,
                weight_decay=weight_decay,
                seed=42
            )

            history = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                initial_lr=lr,
                decay_epochs=decay_epochs,
                verbose=True
            )

            all_results[config_name] = history

            max_val_acc = max(history['val_acc'])
            print("Best val acc for {}: {:.4f}".format(config_name, max_val_acc))

            if max_val_acc > best_val_acc:
                best_val_acc = max_val_acc
                best_config = {
                    'activation': act_name,
                    'lr': lr,
                    'name': config_name
                }

    print("\nBest configuration:")
    print(best_config)
    print("Best validation accuracy:", best_val_acc)

    plot_histories(all_results, decay_epochs=decay_epochs)


    print("\n" + "=" * 70)
    print("Retraining on all 60,000 training images...")
    print("=" * 70)

    X_full_train = X_all[:60000]
    y_full_train = y_all[:60000]

    best_model = MLP(
        layer_sizes=layer_sizes,
        activation=best_config['activation'],
        weight_decay=weight_decay,
        seed=42
    )

    lr = best_config['lr']
    final_epochs = 50

    for epoch in range(final_epochs):
        if epoch in decay_epochs:
            lr = lr / 2.0

        for Xb, yb in get_minibatches(X_full_train, y_full_train, batch_size=batch_size, shuffle=True):
            yb_onehot = one_hot(yb, num_classes=10)

            cache, probs = best_model.forward(Xb)
            grads = best_model.backward(cache, yb_onehot)
            best_model.update_params(grads, lr)

        train_loss, train_acc = best_model.evaluate(X_full_train, y_full_train)
        print("[Full Train] Epoch {:02d} | lr={:.5f} | loss={:.4f}, acc={:.4f}".format(
            epoch + 1, lr, train_loss, train_acc
        ))

    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print("\nFinal Test Accuracy:", test_acc)
    print("Final Test Loss:", test_loss)

    return {
        'all_results': all_results,
        'best_config': best_config,
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc,
        'final_test_loss': test_loss
    }



if __name__ == '__main__':
    train_path = 'mnist_traindata.hdf5'
    test_path = 'mnist_testdata.hdf5'

    results = run_experiments(train_path, test_path)

    print("\n===== Summary =====")
    print("Best config:", results['best_config'])
    print("Best val acc:", results['best_val_acc'])
    print("Final test acc:", results['final_test_acc'])
    print("Final test loss:", results['final_test_loss'])


# Network Configuration
# 
# I used a fully connected multilayer perceptron with architecture 784-256-128-10.
# The input layer has 784 neurons, corresponding to the 28×28 MNIST image pixels.
# There are two hidden layers with 256 and 128 neurons respectively.
# The output layer has 10 neurons, one for each digit class.
# ReLU and tanh were tested as hidden-layer activation functions.
# The output layer uses softmax.

# Batch 
# 
# I used a minibatch size of 100.
# This divides the 50,000 training images evenly into 500 updates per epoch.

# Initial Learning Rates
# 
# I tested three initial learning rates: 0.1, 0.01, and 0.001.
# For each learning rate, I trained the model using both ReLU and tanh activation functions, resulting in 6 configurations in total.
# The learning rate was reduced by a factor of 2 after epoch 20 and epoch 40.

# Parameter Initialization
# 
# I did not initialize weights with zeros.
# For hidden layers using ReLU, I used He initialization.
# For hidden layers using tanh, I used Xavier initialization.
# The output layer was initialized using Xavier initialization.
# All bias vectors were initialized to zero.

# Training and Validation Accuracy Curves
# 
# I trained 6 configurations in total:
# 1. ReLU, learning rate = 0.1
# 2. ReLU, learning rate = 0.01
# 3. ReLU, learning rate = 0.001
# 4. tanh, learning rate = 0.1
# 5. tanh, learning rate = 0.01
# 6. tanh, learning rate = 0.001
# 
# For each configuration, I plotted both training accuracy and validation accuracy versus epoch number.
# The epochs where learning rate decay was applied (epoch 20 and epoch 40) were marked on each plot.
# 
# From the plots, configurations with learning rate 0.1 converged faster and achieved higher validation accuracy.
# The models with learning rate 0.001 improved much more slowly and had lower final accuracy.
# Among all six configurations, ReLU with initial learning rate 0.1 achieved the best validation performance.
# 

# Best Model and Final Test Accuracy
# 
# The best configuration was ReLU with initial learning rate 0.1.
# Its best validation accuracy was 0.9802 (98.02%).
# After selecting this configuration, I retrained the network on all 60,000 training images.
# I then evaluated the final model on the test set.
# The final test accuracy was 0.9797 (97.97%).

# In[ ]:




