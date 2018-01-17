import pandas as pd
import numpy as np
import random

def get_data(fname):
    if fname == 'train':
        train = pd.read_csv("/Users/anshuwang/Documents/Work/kaggle/toxic/data/train.csv")
        return train
    elif fname == 'test':
        test = pd.read_csv("/Users/anshuwang/Documents/Work/kaggle/toxic/data/test.csv")
        return test


def split_data(x, y, test_size=0.2, shuffle=True, random_state=42):
    n = len(x)
    train_idxs, test_idxs = split_data_idx(n, test_size, shuffle, random_state)
    return np.array(x[train_idxs]), np.array(x[test_idxs]), y[train_idxs], y[test_idxs], train_idxs, test_idxs

def split_data_idx(n, test_size=0.2, shuffle=True, random_state=0):
    train_size = 1 - test_size
    idxs = np.arange(n)
    if shuffle:
        random.seed(random_state)
        random.shuffle(idxs)
    return idxs[:int(train_size*n)], idxs[int(train_size*n):]
