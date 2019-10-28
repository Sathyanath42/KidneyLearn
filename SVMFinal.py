import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("Cleankidney.txt")
"""print(data.head())"""

le = preprocessing.LabelEncoder()
age = le.fit_transform(list(data["age"]))
al = le.fit_transform(list(data["al"]))
bu = le.fit_transform(list(data["bu"]))
sc = le.fit_transform(list(data["sc"]))
htn = le.fit_transform(list(data["htn"]))
dm = le.fit_transform(list(data["dm"]))
cls = le.fit_transform(list(data["class"]))
predict = "class"

x = list(zip(age, al, bu, sc, htn, dm))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
names = ["ckd", "notckd"]

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)

for x in range(len(y_pred)):
    print("predicted:", names[y_pred[x]], "Data:", x_test[x], "actual:", names[y_test[x]])
