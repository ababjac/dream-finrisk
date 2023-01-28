#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import pandas as pd
import numpy as np
import sys
import os

from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.impute import KNNImputer
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras as ks
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.linear_model import Lasso
from sksurv.util import Surv


# In[2]:


def fillNA(df, k_neighbors=10, index=[]):
    if not list(index):
        index = df.index
    
    X = df.values
    knn = KNNImputer(n_neighbors=k_neighbors)
    X_imp = knn.fit_transform(X)
    
    df_imp = pd.DataFrame(X_imp, columns=df.columns)
    df_imp.set_index(index, inplace=True)
    
    return df_imp

def minmax_scale(train, test):
    xtrain_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled  

def split_and_scale_data(features, labels, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=5, stratify=labels)
    X_train_scaled, X_test_scaled = minmax_scale(X_train, X_test)
    X_train_scaled.set_index(X_train.index, inplace=True)
    X_test_scaled.set_index(X_test.index, inplace=True)

    return X_train_scaled, X_test_scaled, y_train, y_test

def make_structured_array(event, time):
    return Surv.from_arrays(event, time)

class Autoencoder(ks.models.Model):
    def __init__(self, actual_dim, latent_dim, activation, loss, optimizer):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = ks.Sequential([
        ks.layers.Flatten(),
        ks.layers.Dense(latent_dim, activation=activation),
        ])

        self.decoder = ks.Sequential([
        ks.layers.Dense(actual_dim, activation=activation),
        ])

        self.compile(loss=loss, optimizer=optimizer, metrics=[ks.metrics.BinaryAccuracy(name='accuracy')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_AE(actual_dim=1, latent_dim=100, activation='relu', loss='MAE', optimizer='Adam'):
    return Autoencoder(actual_dim, latent_dim, activation, loss, optimizer)

def run_AE(X_train_scaled, X_test_scaled, param_grid=None):

    if param_grid == None:
        param_grid = {
            'actual_dim' : [len(X_train_scaled.columns)],
            'latent_dim' : [100, 200, 500],
            'activation' : ['relu', 'sigmoid', 'tanh'],
            'loss' : ['MAE', 'binary_crossentropy'],
            'optimizer' : ['SGD', 'Adam']
        }

    model = KerasClassifier(build_fn=create_AE, epochs=10, verbose=0)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=3
    )

    result = grid.fit(X_train_scaled, X_train_scaled, validation_data=(X_test_scaled, X_test_scaled))
    params = grid.best_params_
    autoencoder = create_AE(**params)

    try:
        encoder_layer = autoencoder.encoder
    except:
        exit

    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train.add_prefix('feature_')
    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test.add_prefix('feature_')

    return AE_train, AE_test

def run_LASSO(X_train_scaled, y_train, param_grid = None):
    if param_grid == None:
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

    search = GridSearchCV(estimator = Lasso(),
                          param_grid = param_grid,
                          cv = 5,
                          scoring="neg_mean_squared_error",
                          verbose=3
                          )

    search.fit(X_train_scaled, y_train)
    coefficients = search.best_estimator_.coef_
    importance = np.abs(coefficients)
    keep = np.array(X_train_scaled.columns)[importance != 0]

    return keep


# In[5]:


INPUT_DIR = sys.argv[1]


# In[4]:


reads_train = pd.read_csv(INPUT_DIR+'/train/readcounts_training.csv', index_col=0)
pheno_train = pd.read_csv(INPUT_DIR+'/train/pheno_training.csv', index_col=0)


# In[ ]:


pheno_train = pheno_train.dropna(subset=['Event', 'Event_time']) #should not try to correct these
reads_train = reads_train.T
train_ids = pheno_train.index


# In[ ]:


reads_test = pd.read_csv(INPUT_DIR+'/test/readcounts_test.csv', index_col=0)
pheno_test = pd.read_csv(INPUT_DIR+'/test/pheno_test.csv', index_col=0)


# In[ ]:


pheno_test = pheno_test.dropna(subset=['Event', 'Event_time']) #should not try to correct these
reads_test = reads_test.T
test_ids = pheno_test.index


# In[ ]:


reads_full = pd.concat([reads_train, reads_test])
pheno_full = pd.concat([pheno_train, pheno_test])


# In[ ]:


pheno_full = fillNA(pheno_full, index=pheno_full.index) #fast KNN


# In[ ]:


features = pd.merge(reads_full, pheno_full, left_index=True, right_index=True)
labels = features.pop('Event')
true_labels = labels.copy()
#labels.value_counts() # 0 - 4898, 1 - 445 --> highly imbalanced

X_train, y_train = features.loc[train_ids], labels[train_ids]
X_test, y_test = features.loc[test_ids], labels[test_ids]

X_train, X_test = minmax_scale(X_train, X_test)


# In[ ]:


metadata = [
            'Age', 
            'BodyMassIndex', 
            'BPTreatment', 
            'Smoking', 
            'PrevalentDiabetes', 
            'PrevalentHFAIL', 
            'Event_time', 
            'SystolicBP', 
            'NonHDLcholesterol', 
            'Sex'
           ]

meta_train = X_train[metadata].copy()
rds_train = X_train.drop(metadata, axis=1)

meta_test = X_test[metadata].copy()
rds_test = X_test.drop(metadata, axis=1)


# In[ ]:


AE_train, AE_test = run_AE(rds_train, rds_test)


# In[ ]:


ET = meta_train.pop('Event_time')
important_cols = run_LASSO(meta_train, ET)
important_cols = np.append(important_cols, 'Event_time')


# In[ ]:


meta_train['Event_time'] = ET


# In[ ]:


AE_train_df = AE_train.join(meta_train[important_cols])
AE_train_df = AE_train_df.set_index(train_ids)

AE_test_df = AE_test.join(meta_test[important_cols])
AE_test_df = AE_test_df.set_index(test_ids)


# In[ ]:


df_AE = pd.concat([AE_train_df, AE_test_df])
df_AE['Event'] = true_labels


# In[ ]:


features = df_AE.copy()
times = features.pop('Event_time')
events = features.pop('Event')

train_times = times[train_ids]
train_events = events[train_ids]
y_train = make_structured_array(train_events, train_times)

test_times = times[test_ids]
test_events = events[test_ids]
y_test = make_structured_array(test_events, test_times)

X_train = features.loc[train_ids]
X_test = features.loc[test_ids]


# In[ ]:


est_cph_tree = GradientBoostingSurvivalAnalysis(
    n_estimators=1000, 
    random_state=8993624,
    loss='coxph',
    max_depth=5,
    max_features='sqrt',
    verbose=1
    )
est_cph_tree.fit(X_train, y_train)
est_cph_tree.score(X_test, y_test)


# In[ ]:


OUTPUT_DIR = 'UTK_Bioinformatics_Submission_1/output/'


# In[ ]:


CHECK_FOLDER = os.path.isdir(OUTPUT_DIR)
if not CHECK_FOLDER:
    os.makedirs(OUTPUT_DIR)


# In[ ]:


y_preds = est_cph_tree.predict(X_test)


# In[ ]:


y_preds = normalize(y_preds) #ensure range 0-1


# In[ ]:


data = {'SampleID' : test_ids, 'Score' : y_preds}
out_df = pd.DataFrame(data)
out_df.to_csv(OUTPUT_DIR+'scores.csv', index=False)

