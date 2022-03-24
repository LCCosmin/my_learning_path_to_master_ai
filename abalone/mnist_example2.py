import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime, relu, relu_prime, softplus, softplus_prime
from fc_layer import FCLayer
from losses import mse, mse_prime
from network import Network

file = open("abalone.data")

x_abalone = []
y_abalone = []

# read from file
with file as csv:
    for line in csv.readlines():
        tokens = line.rstrip('\n').split(',')
        if tokens[0] == '':
            break
        
        x_abalone.append((float(tokens[1]), float(tokens[2]), float(tokens[3]), 
                         float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7])))
        
        if tokens[0] == 'M':
            y_abalone.append(0)
        elif tokens[0] == 'F':
            y_abalone.append(1)
        elif tokens[0] == 'I':
            y_abalone.append(2)
        
x_abalone = np.array(x_abalone, dtype=(float)).reshape(-1, 7)
y_abalone = np.array(y_abalone, dtype=(int))

x_train, x_test, y_train, y_test = train_test_split(x_abalone, y_abalone, 
                                                    test_size = 0.2, shuffle=(True))

# normalization theorem
for column in range(7):
    x_train[:, column] = (x_train[:, column] - np.min(x_train[:, column])) / (
        np.max(x_train[:, column]) - np.min(x_train[:, column]))
    
for column in range(7):
    x_test[:, column] = (x_test[:, column] - np.min(x_train[:, column])) / (
        np.max(x_train[:, column]) - np.min(x_train[:, column]))
    
output_sol_max = np.max(y_abalone)

# model

model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape = (7,)),
        tf.keras.layers.Dense(77, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(49, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(14, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3)
    ])



model.compile(optimizer = 'adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])


history = model.fit(x_train, y_train, epochs=100, batch_size =100, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

print("Accuracy :" + str(accuracy))

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'], loc='upper right')
plt.show()
