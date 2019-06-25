
#%%
import pandas as pd
import numpy as np
from keras.utils import to_categorical

train = pd.read_csv('C:/Users/meeta/Desktop/TSWE2019/sign-language-recognition/dataset/sign_mnist_train.csv', header = 0)
test = pd.read_csv('C:/Users/meeta/Desktop/TSWE2019/sign-language-recognition/dataset/sign_mnist_test.csv', header = 0)

y_train = train['label'].values
y_test = test['label'].values

x_train = train.drop(['label'],axis=1)
x_test = test.drop(['label'], axis=1)

x_train = np.array(x_train.iloc[:,:])
# print(x_train[:5])
x_train = np.array([np.reshape(i, (28, 28)) for i in x_train])

x_test = np.array(x_test.iloc[:,:])
x_test = np.array([np.reshape(i, (28, 28)) for i in x_test])
# print(x_test[0])

x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print('Training examples:',x_train.shape[0])
print('Test examples:',x_test.shape[0])

print('X_train shape"',x_train.shape)
print('y_train shape"',y_train.shape)
print('X_test shape"',x_test.shape)
print('y_test shape"',y_test.shape)

x_train = x_train.reshape((27455, 28, 28, 1))
x_test = x_test.reshape((7172, 28, 28, 1))

print(x_train.shape)
print(y_train.shape)
# print(x_train[:3])
# print(y_train[:3])


#%%
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pydot


#%%
def generateModel():    
    model = Sequential()
    model.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = (28,28,1),activation = 'relu',padding = 'same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64,kernel_size = (3,3),padding = 'same',activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(25,activation = 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#%%
classifier = generateModel()


#%%
classifier.summary()


#%%
y_train[0].shape


#%%
classifier.fit(x_train, y_train, batch_size = 100, epochs = 10)


#%%
y_prediction = classifier.predict(x_test)


#%%
y_prediction


#%%



#%%



#%%



#%%



