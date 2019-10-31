import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("Cleankidney.txt")  # This code is for printing the data head
# print(data.head())

le = preprocessing.LabelEncoder()                # Here we pre-process the data into variables
age = le.fit_transform(list(data["age"]))        # for eg false could be 0 and true could be 1
al = le.fit_transform(list(data["al"]))
bu = le.fit_transform(list(data["bu"]))
sc = le.fit_transform(list(data["sc"]))
htn = le.fit_transform(list(data["htn"]))
dm = le.fit_transform(list(data["dm"]))
cls = le.fit_transform(list(data["class"]))
predict = "class"

X = list(zip(age, al, bu, sc, htn, dm))  # We reconstruct the row of input columns
Y = list(cls)  # We reconstruct the row of output columns

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
# We split the available data into training data(10%) and test data (90%)" and assign them variables

model = KNeighborsClassifier(n_neighbors=11)  # Calling KNeighbour function with 11 neighbours

model.fit(x_train, y_train)  # Fitting the training data
acc = model.score(x_test, y_test)  # Checking the accuracy of the fitting
print(acc)

predicted = model.predict(x_test)  # Predicting the output
names = ["ckd", "notckd"]

# CHECKING

for x in range(len(predicted)):  # Array of the predicted values, the actual value and the data
    print("predicted:", names[predicted[x]], "Data:", x_test[x], "actual:", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 11, True)
    print("N: ", n)
