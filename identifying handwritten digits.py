from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

data_set = load_digits()
data = data_set.data
target = data_set.target
images = data_set.images
print('Data Shape', data.shape)
print('Target Shape', target.shape)
print('Images Shape', images.shape)

plt.imshow(images[64], cmap='gray')
print('target:', target[64])
print('Data:', data[64])
# plt.imshow(data[1000].reshape(1, 64), cmap='gray')  # this will print the flatten version of data
# print(images[64])
# print(data[64])

# this data and target will select randomly so whenever we run this every time accuracy ill change
# bcz its selecting randomly
result = train_test_split(data, target, test_size=0.2)
train_data = result[0]
test_data = result[1]
train_target = result[2]
test_target = result[3]
# print('Train target:', train_target)
'''print(train_data.shape)
print(train_target.shape)'''
#  Loading KNN algorithm to Model
model = KNeighborsClassifier()
model.fit(train_data, train_target)
# model.fit(train_data, train_target)
print(test_data.shape)

# applying testing images into the trained model
predicted_target = model.predict(test_data)

acc = accuracy_score(test_target, predicted_target)
print('Accuracy:', acc)
print("Predicted digits:-", predicted_target)
print("Actual digits:-", test_target)

# will save this file named KNN-HANDWRITTEN-DIGITS
joblib.dump(model, 'KNN-HANDWRITTEN-DIGITS.sav')

