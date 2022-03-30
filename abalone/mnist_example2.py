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
        
        if tokens[0] == 'M':
            x_abalone.append((0.0, float(tokens[1]), float(tokens[2]), float(tokens[3]), 
                             float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7])))
        elif tokens[0] == 'F':
            x_abalone.append((1.0, float(tokens[1]), float(tokens[2]), float(tokens[3]), 
                             float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7])))
        elif tokens[0] == 'I':
            x_abalone.append((2.0, float(tokens[1]), float(tokens[2]), float(tokens[3]), 
                             float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7])))
            
        y_abalone.append(tokens[8])
        
x_abalone = np.array(x_abalone, dtype=(float)).reshape(-1, 8)
y_abalone = np.array(y_abalone, dtype=(int))

outputs = np.max(y_abalone)

x_train, x_test, y_train, y_test = train_test_split(x_abalone, y_abalone, 
                                                    test_size = 0.2, shuffle=(True))

# normalization theorem
#for column in range(8):
#    x_train[:, column] = tf.math.divide_no_nan((x_train[:, column] - np.min(x_train[:, column])), (
#        np.max(x_train[:, column]) - np.min(x_train[:, column])))
    
#for column in range(8):
#    x_test[:, column] = tf.math.divide_no_nan((x_test[:, column] - np.min(x_train[:, column])), (
#        np.max(x_train[:, column]) - np.min(x_train[:, column])))

tf.keras.utils.normalize(x_train, order=2)
tf.keras.utils.normalize(x_test, order=2)

# model

model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape = (8,),),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(50, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(75, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.1),
        #kernel_regularizer=tf.keras.regularizers.l2(0.001)
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(outputs + 1 - np.min(y_abalone))
    ])

print(np.any(np.isnan(x_train)))
print(np.any(np.isnan(y_train)))

model.compile(optimizer = 'adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])


history = model.fit(x_train, y_train, epochs=128, batch_size = 128)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

print("Accuracy :" + str(accuracy))

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'], loc='upper right')
plt.show()
