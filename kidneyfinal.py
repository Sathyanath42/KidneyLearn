from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
from sklearn import linear_model, preprocessing
import random
import math
import time
import Import_file as file


class Patient(object):
    featureNames = ('age', 'al', 'bu', 'sc', 'htn', 'dm')

    def __init__(self, age, al, bu, sc, htn, dm, ckd):
        self.age = age
        self.al = al
        self.bu = bu
        self.sc = sc
        self.htn = htn
        self.dm = dm
        self.featureVec = [age, al, bu, sc, htn, dm]
        self.label = ckd

    def distance(self, other):
        return EuclideanDistance(self.featureVec, other.featureVec)


def split(examples):
    randomSamples = random.sample(range(len(examples)), len(examples) // 5)
    train, test = [], []
    for i in range(len(examples)):
        if i in randomSamples:
            test.append(examples[i])
        else:
            train.append(examples[i])
    return train, test


def findKNearest(test_example, train_exampleSet, k=3):
    kNearest, dist_KNearest = [], []
    for i in range(k):
        kNearest.append(train_exampleSet[i])
        dist_KNearest.append(test_example.distance(train_exampleSet[i]))
    maxDist = max(dist_KNearest)
    for e in train_exampleSet[k:]:
        dist = test_example.distance(e)
        if dist < maxDist:
            maxIndex = dist_KNearest.index(maxDist)
            kNearest[maxIndex] = e
            dist_KNearest[maxIndex] = dist
            maxDist = max(dist_KNearest)
    return kNearest, dist_KNearest


def EuclideanDistance(p1, p2):
    dist = 0.0
    for j in range(len(p1)):
        dist += abs(p1[j] - p2[j]) ** 2
    return math.sqrt(dist)


def getKidneyData(filename):
    data = {}
    data['age'], data['al'], data['bu'] = [], [], []
    data['sc'], data['htn'], data['dm'], data['ckd'] = [], [], [], []
    f = open(r'Kidney_Data1.txt')
    line = f.readline()
    while line != '':
        separate = line.split(',')
        data['age'].append(float(separate[0]))
        data['al'].append(float(separate[1]))
        data['bu'].append(float(separate[2]))
        data['sc'].append(float(separate[3]))
        if separate[4] == 'yes':
            data['htn'].append(1)
        else:
            data['htn'].append(0)
        if separate[5] == 'yes':
            data['dm'].append(1)
        else:
            data['dm'].append(0)
        if separate[6] == '1':
            data['ckd'].append("ckd")
        else:
            data['ckd'].append("not ckd")
        line = f.readline()
    return data


def buildKidneyExamples(fileName):
    data = getKidneyData(fileName)
    examples = []
    for i in range(len(data['age'])):
        p = Patient(data['age'][i], data['al'][i], data['bu'][i], data['sc'][i], data['htn'][i], data['dm'][i],
                    data['ckd'][i])
        examples.append(p)
    print('Finished processing', len(examples), 'patients\n')
    return examples


def accuracy(TruePositive, FalsePositive, TrueNegative, FalseNegative):
    denom = TruePositive + TrueNegative + FalsePositive + FalseNegative
    num = TrueNegative + TruePositive
    return num / denom


def sensitivity(TruePositive, FalseNegative):
    try:
        x = TruePositive / (TruePositive + FalseNegative)
        return x
    except ZeroDivisionError:
        return float('nan')


def specificity(TrueNegative, FalsePositive):
    try:
        x = TrueNegative / (TrueNegative + FalsePositive)
        return x
    except ZeroDivisionError:
        return float('nan')


def PositivePredictedValue(TruePositive, FalsePositive):
    try:
        x = TruePositive / (TruePositive + FalsePositive)
        return x
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


def KNearestNeighbours(k=3):
    examples = buildKidneyExamples('Kidney_Data1.txt')
    train, test = split(examples)
    for sample in test:
        nearest, distances = findKNearest(sample, train, k=3)
    print("Object: ", nearest)
    print("Distance: ", distances, "\n")
    return None


def KNearestClassifer3(k=3):
    start = time.time()
    data = file.import_files2()
    kidney = preprocessing.LabelEncoder()
    age = kidney.fit_transform(list(data["Age"]))
    al = kidney.fit_transform(list(data["Albumin (al)"]))
    bu = kidney.fit_transform(list(data["Blood Urea (bu)"]))
    sc = kidney.fit_transform(list(data["Serum Creatinine (sc)"]))
    htn = kidney.fit_transform(list(data["High Blood Pressure (htn)"]))
    dm = kidney.fit_transform(list(data["Diabetes Mellitus (dm)"]))
    cls = kidney.fit_transform(list(data["class"]))
    predict = "class"
    x = list(zip(age, al, bu, sc, htn, dm))
    y = list(cls)
    x_train, x_test = split(x)
    y_train, y_test = split(y)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    names = ["ckd", "notcdk"]
    acc = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:", acc, "\n")
    for i in range(len(predicted)):
        print("Predicted: ", names[predicted[i]], " Data: ", x_test[i], " Actual: ", names[y_test[i]])
    tp, tn, fp, fn = 0, 0, 0, 0
    for j in range(len(predicted)):
        if predicted[j] == y_test[j] == 0:
            tp += 1
        if predicted[j] == y_test[j] == 1:
            tn += 1
        if predicted[j] == 0 and y_test[j] != predicted[j]:
            fp += 1
        if predicted[j] == 1 and y_test[j] != predicted[j]:
            fn += 1
    print("\n" + "Count of true positive=", tp, " Count of true negative=", tn)
    print("Count of false positive=", fp, " Count of false negative=", fn, "\n")
    getPerformance(tp, fp, tn, fn)
    KNearestNeighbours(k=3)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")
    return None


def KNearestClassifer5(k=5):
    start = time.time()
    data = file.import_files2()
    kidney = preprocessing.LabelEncoder()
    age = kidney.fit_transform(list(data["Age"]))
    al = kidney.fit_transform(list(data["Albumin (al)"]))
    bu = kidney.fit_transform(list(data["Blood Urea (bu)"]))
    sc = kidney.fit_transform(list(data["Serum Creatinine (sc)"]))
    htn = kidney.fit_transform(list(data["High Blood Pressure (htn)"]))
    dm = kidney.fit_transform(list(data["Diabetes Mellitus (dm)"]))
    cls = kidney.fit_transform(list(data["class"]))
    predict = "class"
    x = list(zip(age, al, bu, sc, htn, dm))
    y = list(cls)
    x_train, x_test = split(x)
    y_train, y_test = split(y)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    names = ["ckd", "notcdk"]
    for i in range(len(predicted)):
        print("Predicted: ", names[predicted[i]], " Data: ", x_test[i], " Actual: ", names[y_test[i]])
    acc = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:", acc, "\n")
    tp, tn, fp, fn = 0, 0, 0, 0
    for j in range(len(predicted)):
        if predicted[j] == y_test[j] == 0:
            tp += 1
        if predicted[j] == y_test[j] == 1:
            tn += 1
        if predicted[j] == 0 and y_test[j] != predicted[j]:
            fp += 1
        if predicted[j] == 1 and y_test[j] != predicted[j]:
            fn += 1
    print("\n" + "Count of true positive=", tp, " Count of true negative=", tn)
    print("Count of false positive=", fp, " Count of false negative=", fn, "\n")
    getPerformance(tp, fp, tn, fn)
    KNearestNeighbours(k=5)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")
    return None


def KNearestClassifer7(k=7):
    start = time.time()
    data = file.import_files2()
    kidney = preprocessing.LabelEncoder()
    age = kidney.fit_transform(list(data["Age"]))
    al = kidney.fit_transform(list(data["Albumin (al)"]))
    bu = kidney.fit_transform(list(data["Blood Urea (bu)"]))
    sc = kidney.fit_transform(list(data["Serum Creatinine (sc)"]))
    htn = kidney.fit_transform(list(data["High Blood Pressure (htn)"]))
    dm = kidney.fit_transform(list(data["Diabetes Mellitus (dm)"]))
    cls = kidney.fit_transform(list(data["class"]))
    predict = "class"
    x = list(zip(age, al, bu, sc, htn, dm))
    y = list(cls)
    x_train, x_test = split(x)
    y_train, y_test = split(y)
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    names = ["ckd", "notcdk"]
    for i in range(len(predicted)):
        print("Predicted: ", names[predicted[i]], " Data: ", x_test[i], " Actual: ", names[y_test[i]])
    acc = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:", acc, "\n")
    tp, tn, fp, fn = 0, 0, 0, 0
    for j in range(len(predicted)):
        if predicted[j] == y_test[j] == 0:
            tp += 1
        if predicted[j] == y_test[j] == 1:
            tn += 1
        if predicted[j] == 0 and y_test[j] != predicted[j]:
            fp += 1
        if predicted[j] == 1 and y_test[j] != predicted[j]:
            fn += 1
    print("\n" + "Count of true positive=", tp, " Count of true negative=", tn)
    print("Count of false positive=", fp, " Count of false negative=", fn, "\n")
    getPerformance(tp, fp, tn, fn)
    KNearestNeighbours(k=7)
    end = time.time()
    print("Processing Time: ", round(end - start, 6), "seconds","\n")
    return None
