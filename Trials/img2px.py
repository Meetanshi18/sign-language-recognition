import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# plt.subplot(#rows, #cols, position)
# plt.subplot(1, 2, 1)

img = cv2.imread('opencv_frame_0.png', 0)
# img = cv2.imread('1.png')
print(img.shape)
img = img[200:400, 200:]
img = cv2.resize(img, (28,28))
print(img)
img = np.array(img)
print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
plt.imshow(np.array(img), interpolation='nearest')
plt.show()
# ---------------------------------------------------------------------

# import pandas as pd
# train = pd.read_csv('C:/Users/meeta/Desktop/TSWE2019/sign-language-recognition/dataset/sign_mnist_train.csv', header = 0)
# x_train = train.drop(['label'],axis=1)
# x_train = np.array(x_train.iloc[:])
# x_train = np.array([np.reshape(i, (28, 28)) for i in x_train])


# for i in range(5):   
#     # img = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2RGB) 
#     plt.subplot(2, 5, i+1)
#     plt.title(i+1)
#     plt.imshow(x_train[i], interpolation='nearest')
# for i in range(5):
#     plt.subplot(2, 5, 5+i+1)
#     plt.imshow(x_train[i],cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

