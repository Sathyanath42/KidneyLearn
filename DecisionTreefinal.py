import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd
import numpy as np
import time
import pickle
from sklearn import linear_model, preprocessing
from matplotlib import style
import matplotlib.pyplot as pyplot
import Import_file as file


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
    trpo = 0  # false positive and false negative
    trne = 0
    flpo = 0
    flne = 0
    for i in range(len(predic)):
        if (predic[i] == 0) and (tested[i] == 0):
            trpo = trpo + 1

        elif (predic[i] == 0) and (tested[i] == 1):
            flpo = flpo + 1

        elif (predic[i] == 1) and (tested[i] == 1):
            trne = trne + 1

        else:
            flne = flne + 1
    return trpo, trne, flpo, flne


def decisions(n, xtr, ytr, xts):  # function for predicting using different criteria in decision tree
    if n == "0":
        clf = DT_entr(xtr, ytr)
        clf = clf.fit(xtr, ytr)
        ypredict = clf.predict(xts)
    else:
        clf = DT_gini(xtr, ytr)
        clf = clf.fit(xtr, ytr)
        ypredict = clf.predict(xts)

    return ypredict, clf


def plotgraph(p1, p2, dta):
    style.use("ggplot")
    pyplot.scatter(dta[p1], dta[p2])
    pyplot.xlabel(p1)
    pyplot.ylabel(p2)
    pyplot.show()


def realtime_entropy():  # Function for executing decision tree in real time
    kdata = file.import_files2()
    best = 0
    for j in range(1):
        x_train, x_test, y_train, y_test = process_split(kdata, 0.2)
        y_pred, dclass = decisions(0, x_train, y_train, x_test)
        acc = metrics.accuracy_score(y_test, y_pred)  # function for determining accuracy
        if acc > best:
            best = acc
            with open("kidneyentropy.pickle", "wb") as f:
                pickle.dump(dclass, f)

    names = ["ckd", "notckd"]
    print("Accuracy: ", best, "\n")

    tp, tn, fp, fn = metricfunc(y_test, y_pred)

    for x in range(len(y_pred)):
        print("Predicted:", names[y_pred[x]], " Data:", x_test[x], " Actual:", names[y_test[x]])

    print("\n" + "Count of true positive=", tp, " Count of true negative=", tn)
    print("Count of false positive=", fp, " Count of false negative=", fn, "\n")

    # xplot = input("input the x axis")
    # yplot = input("input the y axis")
    # plotgraph(xplot, yplot, kdata)
    return tp, fp, tn, fn


def realtime_gini():  # Function for executing decision tree in real time
    kdata = file.import_files2()
    best = 0
    for j in range(1):
        x_train, x_test, y_train, y_test = process_split(kdata, 0.2)
        y_pred, dclass = decisions(1, x_train, y_train, x_test)
        acc = metrics.accuracy_score(y_test, y_pred)  # function for determining accuracy
        if acc > best:
            best = acc
            with open("kidneygini.pickle", "wb") as f:
                pickle.dump(dclass, f)
    names = ["ckd", "notckd"]
    print("Accuracy: ", best, "\n")
    tp, tn, fp, fn = metricfunc(y_test, y_pred)
    for x in range(len(y_pred)):
        print(" Predicted:", names[y_pred[x]], " Data:", x_test[x], " Actual:", names[y_test[x]])

    print("\n" + "Count of true positive=", tp, " Count of true negative=", tn)
    print("Count of false positive=", fp, " Count of false negative=", fn, "\n")

    # xplot = input("input the x axis")
    # yplot = input("input the y axis")
    # plotgraph(xplot, yplot, kdata)
    return tp, fp, tn, fn


# def saved():  # Function for using the pickle files for decision tree
#     kdata = pd.read_csv("Cleankidney.txt")
#     datahead = list(kdata.columns.values)
#     x_train, x_test, y_train, y_test = process_split(kdata, 0.2)
#     nc = input("Enter your choice of criteria: 0 for entropy, 1 for gini")
#     if not (nc == "1") and not (nc == "0"):
#         print("Invalid entry")
#         exit()
#     if nc == "0":
#         pickle_in = open("kidneyentropy.pickle", "rb")
#     else:
#         pickle_in = open("kidneygini.pickle", "rb")
#     dclass = pickle.load(pickle_in)
#     y_pred = dclass.predict(x_test)
#     names = ["ckd", "notckd"]
#
#     acc = metrics.accuracy_score(y_test, y_pred)  # function for determining accuracy
#     print(acc)
#
#     tp, tn, fp, fn = metricfunc(y_test, y_pred)
#
#     print("count of true positive=", tp, "count of true negative=", tn)
#     print("count of false positive=", fp, "count of false negative=", fn)
#
#     for x in range(len(y_pred)):
#         print("predicted:", names[y_pred[x]], "Data:", x_test[x], "actual:", names[y_test[x]])
#
#     xplot = input("input the x axis")
#     yplot = input("input the y axis")
#     plotgraph(xplot, yplot, kdata)
#     return tp, fp, tn, fn


def accuracy(TruePositive, FalsePositive, TrueNegative, FalseNegative):
    denom = TruePositive + TrueNegative + FalsePositive + FalseNegative
    num = TrueNegative + TruePositive
    return num / denom


def sensitivity(TruePositive, FalseNegative):
    try:
        return TruePositive / (TruePositive + FalseNegative)
    except ZeroDivisionError:
        return float('nan')


def specificity(TrueNegative, FalsePositive):
    try:
        return TrueNegative / (TrueNegative + FalsePositive)
    except ZeroDivisionError:
        return float('nan')


def PositivePredictedValue(TruePositive, FalsePositive):
    try:
        return TruePositive / (TruePositive + FalsePositive)
    except ZeroDivisionError:
        return float('nan')


def NegativePredictedValue(TrueNegative, FalseNegative):
    try:
        return TrueNegative / (TrueNegative + FalseNegative)
    except ZeroDivisionError:
        return float('nan')


def getPerformance(TruePositive, FalsePositive, TrueNegative, FalseNegative):
    Accuracy = accuracy(TruePositive, FalsePositive, TrueNegative, FalseNegative)
    Sensitivity = sensitivity(TruePositive, FalseNegative)
    Specificity = specificity(TrueNegative, FalsePositive)
    ppv = PositivePredictedValue(TruePositive, FalsePositive)
    npv = NegativePredictedValue(TrueNegative, FalseNegative)
    print(' Accuracy =', round(Accuracy, 3))
    print(' Sensitivity =', round(Sensitivity, 3))
    print(' Specificity =', round(Specificity, 3))
    print(' Pos. Pred. Val. =', round(ppv, 3))
    print(' Neg. Pred. Val. =', round(npv, 3), "\n")
    return (Accuracy, Sensitivity, Specificity, ppv, npv)


def print_Entropy():
    start = time.time()
    tp, fp, tn, fn = realtime_entropy()
    getPerformance(tp, fp, tn, fn)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")


def print_Gini():
    start = time.time()
    tp, fp, tn, fn = realtime_gini()
    getPerformance(tp, fp, tn, fn)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")
