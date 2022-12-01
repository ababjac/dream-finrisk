import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer

from chardet.universaldetector import UniversalDetector
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso

from sksurv.util import Surv


def make_structured_array(event, time):
    return Surv.from_arrays(event, time)

#--------------------------------------------------------------------------------------------------#

def detect_encoding(file):
    detector = UniversalDetector()
    detector.reset()
    with open(file, 'rb') as f:
        for row in f:
            detector.feed(row)
            if detector.done: break

    detector.close()
    return detector.result['encoding']

#--------------------------------------------------------------------------------------------------#

def standard_scale(train, test):
    xtrain_scaled = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(StandardScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

def minmax_scale(train, test):
    xtrain_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled  

#--------------------------------------------------------------------------------------------------#

def clean_data(data):
    remove = [col for col in data.columns if data[col].isna().sum() != 0]
    return data.loc[:, ~data.columns.isin(remove)] #this gets rid of remaining NA

#--------------------------------------------------------------------------------------------------#

def split_and_scale_data(features, labels, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=5, stratify=labels)
    X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)
    X_train_scaled.set_index(X_train.index, inplace=True)
    X_test_scaled.set_index(X_test.index, inplace=True)

    return X_train_scaled, X_test_scaled, y_train, y_test

#--------------------------------------------------------------------------------------------------#

def perform_SMOTE(X, y, k_neighbors=5, random_state=1982):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm

#--------------------------------------------------------------------------------------------------#

def write_list_to_file(filename, l):
    file = open(filename, 'w')

    for elem in l:
        file.write(elem)
        file.write('\n')

    file.close()


#--------------------------------------------------------------------------------------------------#

def fillNA(df, k_neighbors=10, index=[]):
    if not list(index):
        index = df.index
    
    X = df.values
    knn = KNNImputer(n_neighbors=k_neighbors)
    X_imp = knn.fit_transform(X)
    
    df_imp = pd.DataFrame(X_imp, columns=df.columns)
    df_imp.set_index(index, inplace=True)
    
    return df_imp
