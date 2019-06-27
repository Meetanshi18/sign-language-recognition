import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
# print(digits.data)
# print(digits.target)
# print(digits.data[-6])
clf = svm.SVC(gamma = 0.0001, C=100)

print(len(digits.data))

x,y = digits.data[:-10], digits.target[:-10]
# print(x[:5])
print(y[:5])

# clf.fit(x,y)

print('Prediction: ', clf.predict(digits.data[-1:]))
# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()