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
print(x_test[0])