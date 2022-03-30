import numpy as np
from sklearn.model_selection import train_test_split

from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from fc_layer import FCLayer
from losses import mse, mse_prime
from network import Network

x_iris = []
y_iris = []
with open('iris.data', 'rt') as csv:
    for line in csv.readlines():
        tokens = line.rstrip('\n').split(',')
        if tokens[0] == '':
            break
        x_iris.append((float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])))
        if tokens[4] == 'Iris-setosa':
            y_iris.append(0)  # clasa 0 prin conventie
        if tokens[4] == 'Iris-versicolor':
            y_iris.append(1)  # clasa 1
        if tokens[4] == 'Iris-virginica':
            y_iris.append(2)  # clasa 2
            

x_iris = np.array(x_iris, dtype=float).reshape((-1, 4))
y_iris = np.array(y_iris, dtype=int)



n_samples = min(len(x_iris), len(y_iris))

x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris, test_size=0.2, shuffle=True)

# formula de normalizare
# mult mai usor cu numpy
for cls in range(4):
    x_train[:, cls] = (x_train[:, cls] - np.min(x_train[:, cls])) / (np.max(x_train[:, cls]) - np.min(x_train[:, cls]))

for cls in range(4):
    x_test[:, cls] = (x_test[:, cls] - np.min(x_train[:, cls])) / (np.max(x_train[:, cls]) - np.min(x_train[:, cls]))

x_train = x_train.reshape((-1, 1, 4))
y_train_cat = []
for y in y_train:
    temp = np.zeros(3)
    temp[y] = 1
    y_train_cat.append(temp)
y_train_cat = np.array(y_train_cat).reshape((-1, 3))


net = Network()
net.add(FCLayer(4, 10))  # 4 clase de input
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(10, 10))  # 10 mi se pare rezonabil pentru hidden
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(10, 3))  # 3 clase de output
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)


net.fit(x_train, y_train_cat, epochs=100, learning_rate=0.1)

out = net.predict(x_test)
out = np.array(out, dtype=float).reshape((-1, 3))
print('Predicted values:')
print(out)
print(np.argmax(out, axis=1))
print('True values:')
print(y_test)
# da prea bine :)...
