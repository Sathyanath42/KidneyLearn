import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

def process_split(dt, ts):  #function for fitting the values and then splitting the values into a training and test set
    dtc = list(dt.columns.values)
    label = preprocessing.LabelEncoder()
    for i in dtc:
        dt[str(i)] = label.fit_transform(list(dt[str(i)]))
    xc = dt.values[:, 0:6]
    yc = dt.values[:, 6]
    xt, xts, yt, yts = sklearn.model_selection.train_test_split(xc, yc, test_size=ts)
    return xt, xts, yt, yts

def DT_gini(xg, yg): #gini classifier function
    clf_g = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5)
    clf_g.fit(xg, yg)
    return clf_g

def DT_entr(xe, ye): #entropy classifier function
    clf_e = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_e.fit(xe, ye)
    return clf_e

kdata = pd.read_csv("Cleankidney.txt")  
datahead = list(kdata.columns.values)

x_train, x_test, y_train, y_test = process_split(kdata, 0.2)

clf_er = DT_entr(x_train, y_train)
clf_er = clf_er.fit(x_train, y_train)
y_pred = clf_er.predict(x_test)


# clf_gi = DT_gini(x_train,y_train)
# clf_gi = clf_gi.fit(x_train, y_train)
# y_pred = clf_gi.predict(x_test)

# clf = DecisionTreeClassifier()
# clf = clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)

names = ["ckd", "notckd"]

acc = metrics.accuracy_score(y_test, y_pred) #printing the accuracy
print(acc)

for x in range(len(y_pred)):  #printing out the expected and predicted values
    print("predicted:", names[y_pred[x]], "Data:", x_test[x], "actual:", names[y_test[x]])

