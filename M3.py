'''Classification Analysis (Any one)
A. Implementation of Support Vector Machines (SVM) for classifying images of handwritten digits into their respective numerical classes (0 to 9).'''


# Import Libraries
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

dataset = load_digits()
dataset.data

dataset.target
dataset.data.shape

dataset.target.shape
dataset.images.shape

dataset.images

df2 = dataset.images.reshape(1,-1)
df2.shape

len(dataset.images)

# Check the nth image in the dataset
n = 2

plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]

X = dataset.images.reshape((len(dataset.images), -1))
X.shape

Y =  dataset.target
Y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
print(y_train.shape)
print(X_test.shape)

from sklearn import svm
model = svm.SVC(gamma=0.001)
model.fit(X_train,y_train)

n = 1
result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n],cmap=plt.cm.gray_r,interpolation='nearest')
print(result)

print("\n")

plt.axis("off")
plt.title('%i' %result)
plt.show()

# Set value of nth index here
nth = int(input("Enter index: "))

while (nth < 0 or nth > 1797):
    print("invlaid")
    nth = int(input("Enter index: "))

plt.imshow(dataset.images[nth],cmap=plt.cm.gray_r,interpolation='nearest')
plt.axis("off")

plt.show()

# Predicted value here
predicted_nth = model.predict(dataset.images[nth].reshape((1,-1)))
print("Predicted value: ", predicted_nth)

Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of model is {0}%".format(accuracy_score(y_test,Y_pred)*100))

model1 = svm.SVC(kernel="linear")
model13 = svm.SVC(gamma=0.001)
model14 = svm.SVC(gamma=0.001, C=8.1)

model1.fit(X_train, y_train)
model13.fit(X_train, y_train)
model14.fit(X_train, y_train)

Y_predModel1 = model1.predict(X_test)
Y_predMode13 = model13.predict(X_test)
Y_predMode14 = model14.predict(X_test)

print("Accuracy of Model 1: {:.2f}%".format(accuracy_score(y_test, Y_predModel1) * 100))
print("Accuracy of Model 3: {:.2f}%".format(accuracy_score(y_test, Y_predMode13) * 100))
print("Accuracy of Model 4: {:.2f}%".format(accuracy_score(y_test, Y_predMode14) * 100))