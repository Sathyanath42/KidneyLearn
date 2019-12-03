import pandas as pd



def import_files():
    a = input("Input the file name here: ")
    file_w = open("file_name.txt","w")
    file_w.write(a)
    kdata = pd.read_csv(a)
    datahead = list(kdata.columns.values)
    print(datahead)
    return a


def import_files2():
    file_r = open("file_name.txt", "r")
    # print(file_r.readable())
    b = file_r.readline()
    kdata = pd.read_csv(b)
    return kdata


