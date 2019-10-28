import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("Cleankidney.txt")
"""print(data.head())"""

le = preprocessing.LabelEncoder()   """Pre processing all the values into integers"""
age = le.fit_transform(list(data["age"]))
al = le.fit_transform(list(data["al"]))
bu = le.fit_transform(list(data["bu"]))
sc = le.fit_transform(list(data["sc"]))
htn = le.fit_transform(list(data["htn"]))
dm = le.fit_transform(list(data["dm"]))
cls = le.fit_transform(list(data["class"]))
predict = "class"

X = list(zip(age, al, bu, sc, htn, dm))  "Combining all the preprocessed values into a list"
Y = list(cls)                             """List of all the test values"""

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
"""Making a training set of 90% of the data and the rrest as a testing set"""

model = KNeighborsClassifier(n_neighbors=11)  """Calling KNeighbour function with 11 neighbours"""

model.fit(x_train, y_train)  """Fitting the training data """
acc = model.score(x_test, y_test)  """checking the accuracy of the fitting"""
print(acc)

predicted = model.predict(x_test)  """predicting the output"""
names = ["ckd", "notckd"]

"""CHECKING"""

for x in range(len(predicted)):  """Array of the predicted values, the actual value and the data"""
    print("predicted:", names[predicted[x]], "Data:", x_test[x], "actual:", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 11, True)
    print("N: ", n)