import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model, preprocessing


def process_split(dt, ts):  # function for processing the values and splitting into training and testing sets
    dtc = list(dt.columns.values)
    label = preprocessing.LabelEncoder()
    for i in dtc:
        dt[str(i)] = label.fit_transform(list(dt[str(i)]))
    xc = dt.values[:, 0:6]
    yc = dt.values[:, 6]
    xt, xts, yt, yts = sklearn.model_selection.train_test_split(xc, yc, test_size=ts)
    return xt, xts, yt, yts


def DT_gini(xg, yg):  # function for criteria as gini
    clf_g = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_g.fit(xg, yg)
    return clf_g


def DT_entr(xe, ye):  # function for criteria as entropy
    clf_e = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_e.fit(xe, ye)
    return clf_e


def metricfunc(tested, predic):  # function for counting the number of true positive, true negative,
    trpo = 0                     # false positive and false negative
    trne = 0
    flpo = 0
    flne = 0
    for i in range(len(predic)):
        if (predic[i] == 1) and (tested[i] == 1):
            trpo = trpo + 1

        elif (predic[i] == 1) and (tested[i] == 0):
            flpo = flpo + 1

        elif (predic[i] == 0) and (tested[i] == 0):
            trne = trne + 1

        else:
            flne = flne + 1
    return trpo, trne, flpo, flne


def decisions(n, xtr, ytr, xts):  # function for predicting using different criteria in decision tree
    if (n == 0):
        clf = DT_gini(xtr, ytr)
        clf = clf.fit(xtr, ytr)
        ypredict = clf.predict(xts)
    elif (n == 1):
        clf = DT_entr(x_train, y_train)
        clf = clf.fit(x_train, y_train)
        ypredict = clf.predict(x_test)
    else:
        clf = DecisionTreeClassifier()
        clf = clf.fit(x_train, y_train)
        ypredict = clf.predict(x_test)

    return ypredict, clf


kdata = pd.read_csv("Cleankidney.txt")
datahead = list(kdata.columns.values)

# nc = input("Enter your choice of criteria: 0 for gini, 1 for entropy, 2 for basic decision tree")
# best = 0
x_train, x_test, y_train, y_test = process_split(kdata, 0.2)
# y_pred, dclass = decisions(nc, x_train, y_train, x_test)

# for t in range(30):
#     acc = metrics.accuracy_score(y_test, y_pred)
#     while acc > best:
#        best = acc
#        if nc == 0:
#            with open("kidneygini.pickle", "wb") as f:
#                pickle.dump(dclass, f)
#        elif nc == 1:
#            with open("kidneyentropy.pickle", "wb") as f:
#                pickle.dump(dclass, f)
#        else :
#            with open("kidneybasic.pickle", "wb") as f:
#                pickle.dump(dclass, f)



pickle_in = open("kidneyentropy.pickle", "rb")
dclass = pickle.load(pickle_in)
y_pred = dclass.predict(x_test)

#
names = ["ckd", "notckd"]
#
# acc = metrics.accuracy_score(y_test, y_pred)  # function for determining accuracy
# print(acc)
#
# tp, tn, fp, fn = metricfunc(y_test, y_pred)
#
# print("count of true positive=", tp, "count of true negative=", tn)
# print("count of false positive=", fp, "count of false negative=", fn)
#
for x in range(len(y_pred)):
    print("predicted:", names[y_pred[x]], "Data:", x_test[x], "actual:", names[y_test[x]])
