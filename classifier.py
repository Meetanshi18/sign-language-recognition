import pandas as pd
import numpy as np

train = pd.read_csv('dataset/sign_mnist_train.csv', header = 0)
test = pd.read_csv('dataset/sign_mnist_test.csv', header = 0)

y_train = train['label'].values
y_test = test['label'].values

# print(y_test[:10])
# print(y_train[:10])

x_train = train.drop(['label'],axis=1)
x_test = test.drop(['label'], axis=1)

# print(x_train[:5])

x_train = np.array(x_train.iloc[:])
# print(x_train[:5])
# print(x_train[0][:10])
x_train = np.array([np.reshape(i, (28, 28)) for i in x_train])
# print(x_train[0])

x_test = np.array(x_test.iloc[:])
x_test = np.array([np.reshape(i, (28, 28)) for i in x_test])
# print(x_test[0])

num_classes = 26
# print(y_train.shape)
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)
# print(y_train.shape)

y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

print('Training examples:',x_train.shape[0])
print('Test examples:',x_test.shape[0])

print('X_train shape"',x_train.shape)
print('y_train shape"',y_train.shape)
print('X_test shape"',x_test.shape)
print('y_test shape"',y_test.shape)

x_train = x_train.reshape((27455, 28, 28, 1))
x_test = x_test.reshape((7172, 28, 28, 1))