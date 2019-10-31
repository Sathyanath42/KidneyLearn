import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("Cleankidney.txt")           # This code is for printing the data head
# print(data.head())

le = preprocessing.LabelEncoder()               # Here we pre-process the data into variables for
age = le.fit_transform(list(data["age"]))       # eg false could be 0 and true could be 1
al = le.fit_transform(list(data["al"]))
bu = le.fit_transform(list(data["bu"]))
sc = le.fit_transform(list(data["sc"]))
htn = le.fit_transform(list(data["htn"]))
dm = le.fit_transform(list(data["dm"]))
cls = le.fit_transform(list(data["class"]))
predict = "class"

x = list(zip(age, al, bu, sc, htn, dm))        # We reconstruct the row of input columns
y = list(cls)                                  # We reconstruct the row of output columns

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# We split the available data into training data(20%) and test data (80%)" and assign them variables

clf = svm.SVC(kernel="linear", C=2)
# Here we assign the kernel as linear, other kernels are possible

clf.fit(x_train, y_train)
# We fit the training data as per the svm classifier

y_pred = clf.predict(x_test)
# Predict the response for test dataset

names = ["ckd", "notckd"]
# two possible outputs: checked and not checked

acc = metrics.accuracy_score(y_test, y_pred)
# we measure the accuracy by comparing the predicted values to the expected values
print(acc)

for x in range(len(y_pred)):
    print("predicted:", names[y_pred[x]], "Data:", x_test[x], "actual:", names[y_test[x]])
# Here we print out the predicted values against the expected values
