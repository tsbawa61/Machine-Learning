#%matplotlib inline

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


digits_data = load_digits()

digits = digits_data.data
targets = digits_data.target
a_digit = np.split(digits[0], 8)
plt.title('digits[0]')

plt.imshow(a_digit, cmap='gray')
plt.show()

# SPLIT training data and test data
x_train, x_test, y_train, y_test = train_test_split(digits, targets, test_size=0.25)
print (x_train.shape, x_test.shape)
from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression()
logistic_reg.fit(x_train, y_train)
print(x_test[2].shape)
predict_data = x_test[2].reshape(1, -1)
print(predict_data)
plt.imshow(np.split(x_test[2], 8), cmap='gray')
plt.title('x_test[2]')

predicted=logistic_reg.predict(x_test[4].reshape(1, -1))
print(predicted)
print(predicted.shape)
plt.show()

predict_data = x_test[17].reshape(1, -1)
print(predict_data)
plt.imshow(np.split(x_test[17], 8), cmap='gray')
plt.title('x_test[17]')
predicted=logistic_reg.predict(x_test[17].reshape(1, -1))
print(predicted)
print(predicted.shape)
plt.show()

import cv2 
  
# Read RGB image
fileName='digit_four_1.jpg'
img = cv2.imread(fileName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(gray)
  
cv2.imshow('image', img)  
  
custom_pixels = list(gray)
corr_pixels = []

# convert pixel data to fit training data format (swap grey values)
for row in custom_pixels:
    for x in row:
        x = 255 - x
        corr_pixels.append(x)
print(corr_pixels)
test_set = np.array(corr_pixels)
print(test_set)
a_digit = np.split(test_set, 8)
plt.title(fileName)
plt.imshow(a_digit, cmap='gray')


predicted=logistic_reg.predict(test_set.reshape(1, -1))
print(predicted)
print(predicted.shape)

plt.show()
cv2.imshow(fileName, gray)  


# Destroying present windows on screen 
cv2.destroyAllWindows()  


