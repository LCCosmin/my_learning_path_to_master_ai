import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split

from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from fc_layer import FCLayer
from losses import mse, mse_prime
from network import Network


#read from file
file = open("iris.txt", "r")

all_data = []
correct_val = []

with open('iris.data', 'rt') as csv:
    for line in csv.readlines():
        tokens = line.rstrip('\n').split(',')
        if tokens[0] == '':
            break
        all_data.append((float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])))
        if tokens[4] == 'Iris-setosa':
            correct_val.append(0) 
        if tokens[4] == 'Iris-versicolor':
            correct_val.append(1)
        if tokens[4] == 'Iris-virginica':
            correct_val.append(2) 
            


all_data = np.array(all_data, dtype=float).reshape((-1, 4))
correct_val = np.array(correct_val, dtype=int) 

x_train, x_test, y_train, y_test = train_test_split(all_data, correct_val, test_size = 0.2, shuffle=True)


i = 0

for i in range(4):
    x_train[:, i] = (x_train[:, i] - np.min(x_train[:, i])) / (np.max(x_train[:, i]) - np.min(x_train[:, i]))
    
for i in range(4):
    x_test[:, i] = (x_test[:, i] - np.min(x_train[:, i])) / (np.max(x_train[:, i]) - np.min(x_train[:, i]))

x_train = x_train.reshape((-1,1,4))

y_train_cat = []
for y in y_train:
    temp = np.zeros(3)
    temp[y] = 1
    y_train_cat.append(temp)
y_train_cat = np.array(y_train_cat).reshape((-1, 3))


net = Network()
net.add(FCLayer(2 * 2, 25))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(25, 10))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(10, 3))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

v = []
v = net.fit(x_train, y_train_cat, epochs=100, learning_rate=0.01)

out = net.predict(x_test)
out = np.array(out, dtype=float).reshape((-1, 3))

print('Predicted values:')
print(out)
print(np.argmax(out, axis=1))
print('True values:')
print(y_test)


i = 0
c = 0

for i in range(len(np.argmax(out, axis=1))):
    if np.argmax(out, axis = 1)[i] == y_test[i]:
        c = c+1
        
print("Procentage " + str(c * 100 / len(y_test)))
    
plt.plot(v)