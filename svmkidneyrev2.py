import sklearn
import time
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
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


def kidneyfitsvm():
    mostaccurate = 0
    kdata = file.import_files2()
    for j in range(1):
        x_train, x_test, y_train, y_test = process_split(kdata, 0.2)
        fitsvm = svm.SVC(kernel="linear", C=2)
        fitsvm.fit(x_train, y_train)
        y_pred = fitsvm.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        if acc > mostaccurate:
            mostaccurate = acc
            with open("svmkidney.pickle", "wb") as f:
                pickle.dump(fitsvm, f)
    print("Accuracy:", mostaccurate, "\n")
    pickle_in = open("svmkidney.pickle", "rb")
    fitsvm = pickle.load(pickle_in)
    predicted = fitsvm.predict(x_test)
    names = ["ckd", "notcdk"]
    for x in range(len(predicted)):
        print("Predicted:", names[predicted[x]],predicted[x], " Data:", x_test[x], " Actual:", names[y_test[x]])
    return x_train, x_test, y_train, y_test, predicted


def kidneyfitsvm_rbf():
    mostaccurate = 0
    kdata = file.import_files2()
    for j in range(5):
        x_train, x_test, y_train, y_test = process_split(kdata, 0.2)
        fitsvm = svm.SVC(kernel="rbf", C=2, gamma='auto')
        fitsvm.fit(x_train, y_train)
        y_pred = fitsvm.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        if acc > mostaccurate:
            mostaccurate = acc
            with open("svmkidneyrbf.pickle", "wb") as f:
                pickle.dump(fitsvm, f)
    print("Accuracy:", mostaccurate, "\n")
    pickle_in = open("svmkidneyrbf.pickle", "rb")
    fitsvm = pickle.load(pickle_in)
    predicted = fitsvm.predict(x_test)
    names = ["ckd", "notcdk"]
    for x in range(len(predicted)):
        print("Predicted:", names[predicted[x]], " Data:", x_test[x], " Actual:", names[y_test[x]])
    return x_train, x_test, y_train, y_test, predicted


def kidneyaccuracy(x, y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(len(y)):
        if x[j] == y[j] == 0:
            tp += 1
        if x[j] == y[j] == 1:
            tn += 1
        if x[j] == 0 and y[j] != x[j]:
            fp += 1
        if x[j] == 1 and y[j] != x[j]:
            fn += 1
    print("\n" + "Count of true positive=", tp, " Count of true negative=", tn)
    print("Count of false positive=", fp, " Count of false negative=", fn, "\n")
    return tp, tn, fp, fn


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


def print_svm():
    start = time.time()
    x_train, x_test, y_train, y_test, predicted = kidneyfitsvm()
    tp, tn, fp, fn = kidneyaccuracy(predicted, y_test)
    getPerformance(tp, fp, tn, fn)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")


def print_rbf():
    start = time.time()
    x_train, x_test, y_train, y_test, predicted = kidneyfitsvm_rbf()
    tp, tn, fp, fn = kidneyaccuracy(predicted, y_test)
    getPerformance(tp, fp, tn, fn)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")
