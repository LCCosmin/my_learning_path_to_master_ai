import numpy as np
import tensorflow as tf

def softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def softplus_prime(x): 
    return np.divide(1.,1.+np.exp(-x))

def relu(x):
    return tf.keras.activations.relu(x).numpy()

def relu_prime(x):
    return (x>0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
