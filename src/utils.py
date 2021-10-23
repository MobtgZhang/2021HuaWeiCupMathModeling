import numpy as np
import pandas as pd

# peason
def Peason(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    x = x - x.mean()
    y = y - y.mean()
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
# Jaccard
def Jaccard(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    a = x.dot(y)
    b = np.linalg.norm(x)
    c = np.linalg.norm(y)
    return a/(np.power(b,2)+np.power(c,2)-a)
# cosine
def Cosine(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
def split_train_test(root_file,raw=True,percemtage = 0.75):
    outdata = pd.read_excel(root_file,sheet_name="train")
    names = outdata.iloc[:, 0]
    y_value = outdata.iloc[:, 2]
    x_value = outdata.iloc[:, 3:]
    if raw:
        return x_value.values,y_value.values[:, np.newaxis]
    else:
        length = len(outdata)
        train_len = int(percemtage*length)
        train_x_names = names[:train_len]
        train_x = x_value[:train_len]
        train_y = y_value[:train_len]

        test_x_names = names[train_len:]
        test_x = x_value[train_len:]
        test_y = y_value[train_len:]
        return (train_x.values,train_y.values[:, np.newaxis]),\
               (test_x.values,test_y.values[:, np.newaxis])
