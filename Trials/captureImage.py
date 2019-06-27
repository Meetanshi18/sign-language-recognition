import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------Capturing Image---------------------------------
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

# ----------------------------------Processing the image------------------------------------

img = cv2.imread('opencv_frame_0.png', 0)
print(img.shape)
img = img[150:350, 150:350]
resultImg = img
img = cv2.resize(img, (28,28))
print(img)
img = np.array(img)
print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# plt.imshow(np.array(img), interpolation='nearest')
# plt.show()

sample = img

# -----------------------------------Sent for Prediction--------------------------------------

from sklearn.externals import joblib

classifier = joblib.load('saved_classifier.pkl')

from keras.layers import Dense
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pydot

sample = sample.reshape((1,28,28,1))
res = classifier.predict(sample)
res = list(res[0])
print(res)

mx = max(res)
alphabet = str(res.index(mx))
print(res.index(mx))

# resultImg = np.zeros((512,512,3), np.uint8)
cv2.putText(resultImg, alphabet , (230, 50), cv2.FONT_HERSHEY_COMPLEX , 0.8, (255, 255, 255), 2, cv2.LINE_AA)

plt.imshow(np.array(resultImg), interpolation='nearest')
plt.show()