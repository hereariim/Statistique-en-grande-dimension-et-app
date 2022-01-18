#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 20:17:31 2022

@author: herearii
"""

import numpy as np
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import scipy as sp
from sklearn.linear_model import LinearRegression
#import autosklearn.regression
from lifelines import CoxPHFitter

def premier():
    archive_train = np.load('/home/herearii/Documents/Owkin/x_train/images/patient_002.npz')
    scan = archive_train['scan']
    mask = archive_train['mask']
    
    radiomic_train = pd.read_csv('/home/herearii/Documents/Owkin/x_train/features/radiomics.csv')
    
    clinical_train = pd.read_csv('/home/herearii/Documents/Owkin/x_train/features/clinical_data.csv')
    
    files = listdir('/home/herearii/Documents/Owkin/x_train/images')
    
    name_train = []
    scan_train = []
    mask_train = []
    for file in files:
        path = '/home/herearii/Documents/Owkin/x_train/images/'+file
        archive_train = np.load(path)
        name_train.append(file[8:11])
        scan_train.append(archive_train['scan'])
        mask_train.append(archive_train['mask'])
    image_train = pd.DataFrame({'PatientID':name_train,'CT scan':scan_train,'mask':mask_train})
    
    r0=radiomic_train.iloc[[0]].values.tolist()
    r1=radiomic_train.iloc[[1]].values.tolist()
    
    r0[0][0]=r1[0][0]
    
    
    radiomic_train = radiomic_train.set_axis(r0[0],axis=1)
    
    radiomic_train = radiomic_train.drop([0,1])
    
    radiomic_train.head()
    
    
    IDcl = clinical_train['PatientID'].tolist()
    for k in range(len(IDcl)):
        identity = str(IDcl[k])
        if len(identity)==1:
            IDcl[k]='00'+identity
        elif len(identity)==2:
            IDcl[k]='0'+identity
        else:
            IDcl[k]=identity
    
    clinical_train['PatientID']=IDcl
    clinical_train.head()
    
    
    clinical_train['Histology']=clinical_train['Histology'].fillna('inconnu')
    clinical_train['Histology']=clinical_train['Histology'].str.lower()
    clinical_train['Histology']=clinical_train['Histology'].replace(['nsclc nos (not otherwise specified)'],'nos')
    temp=pd.get_dummies(clinical_train['Histology'])
    temp.index = clinical_train.index
    temp = temp.astype('int64')
    temp.dtypes
    clinical_train = pd.merge(clinical_train,temp,left_index=True, right_index=True)
    clinical_train.pop('Histology')
    
    
    moy = clinical_train['age'].mean(skipna=True)
    clinical_train['age']=clinical_train['age'].fillna(moy)
    
    
    sansID = radiomic_train[list(radiomic_train.columns)[1:]]
    
    sansID = sansID.astype(float)
    
    sansID.describe()
    
    files = listdir('/home/herearii/Documents/Owkin/x_test/images')
    
    name_test = []
    scan_test = []
    mask_test = []
    for file in files:
        path = '/home/herearii/Documents/Owkin/x_test/images/'+file
        archive_test = np.load(path)
        name_test.append(file[8:11])
        scan_test.append(archive_test['scan'])
        mask_test.append(archive_test['mask'])
    image_test = pd.DataFrame({'PatientID':name_test,'CT scan':scan_test,'mask':mask_test})
    
    
    radiomic_test = pd.read_csv('/home/herearii/Documents/Owkin/x_test/features/radiomics.csv')
    
    clinical_test = pd.read_csv('/home/herearii/Documents/Owkin/x_test/features/clinical_data.csv')
    
    
    IDcl = clinical_test['PatientID'].tolist()
    for k in range(len(IDcl)):
        identity = str(IDcl[k])
        if len(identity)==1:
            IDcl[k]='00'+identity
        elif len(identity)==2:
            IDcl[k]='0'+identity
        else:
            IDcl[k]=identity
    
    clinical_test['PatientID']=IDcl
    clinical_test.head()
    
    
    clinical_test['Histology']=clinical_test['Histology'].fillna('inconnu')
    clinical_test['Histology']=clinical_test['Histology'].str.lower()
    clinical_test['Histology']=clinical_test['Histology'].replace(['nsclc nos (not otherwise specified)'],'nos')
    temp=pd.get_dummies(clinical_test['Histology'])
    temp.index = clinical_test.index
    temp = temp.astype('int64')
    clinical_test = pd.merge(clinical_test,temp,left_index=True, right_index=True)
    clinical_test.pop('Histology')
    moy = clinical_test['age'].mean(skipna=True)
    clinical_test['age']=clinical_test['age'].fillna(moy)
    
    
    r0=radiomic_test.iloc[[0]].values.tolist()
    r1=radiomic_test.iloc[[1]].values.tolist()
    
    r0[0][0]=r1[0][0]
    
    
    radiomic_test = radiomic_test.set_axis(r0[0],axis=1)
    
    radiomic_test = radiomic_test.drop([0,1])
    
    radiomic_test.head()
    
    
    
    radiomic_train
    radiomic_train.index=radiomic_train['PatientID']
    radiomic_train.pop('PatientID')
    radiomic_train
    
    
    radiomic_test.index=radiomic_test['PatientID']
    radiomic_test.pop('PatientID')
    
    
    radiomic_test = radiomic_test.astype('float')
    radiomic_train = radiomic_train.astype('float')
    
    
    clinical_train.index=clinical_train['PatientID']
    clinical_train.pop('PatientID')
    
    clinical_test.index=clinical_test['PatientID']
    clinical_test.pop('PatientID')
    
    image_test.index=image_test['PatientID']
    image_test.pop('PatientID')
    
    
    x_train = pd.merge(radiomic_train,clinical_train,left_index=True, right_index=True)
    x_test = pd.merge(radiomic_test,clinical_test,left_index=True, right_index=True)
    
    
    y_train = pd.read_csv('/home/herearii/Documents/Owkin/y_train.csv')
    y_train.index = y_train['PatientID']
    y_train.pop('PatientID')
    
    
    x_train.pop('SourceDataset')
    
    x_test.pop('SourceDataset')
    return x_train,x_test,y_train
    
x_train,x_test,y_train = premier()

def deuxieme():
    archive_train = np.load('/home/herearii/Documents/Owkin/x_train/images/patient_002.npz')
    scan = archive_train['scan']
    mask = archive_train['mask']
    
    radiomic_train = pd.read_csv('/home/herearii/Documents/Owkin/x_train/features/radiomics.csv')
    
    clinical_train = pd.read_csv('/home/herearii/Documents/Owkin/x_train/features/clinical_data.csv')
    
    files = listdir('/home/herearii/Documents/Owkin/x_train/images')
    
    name_train = []
    scan_train = []
    mask_train = []
    for file in files:
        path = '/home/herearii/Documents/Owkin/x_train/images/'+file
        archive_train = np.load(path)
        name_train.append(file[8:11])
        scan_train.append(archive_train['scan'])
        mask_train.append(archive_train['mask'])
    image_train = pd.DataFrame({'PatientID':name_train,'CT scan':scan_train,'mask':mask_train})
    
    r0=radiomic_train.iloc[[0]].values.tolist()
    r1=radiomic_train.iloc[[1]].values.tolist()
    
    r0[0][0]=r1[0][0]
    
    
    radiomic_train = radiomic_train.set_axis(r0[0],axis=1)
    
    radiomic_train = radiomic_train.drop([0,1])
    
    radiomic_train.head()
    
    
    IDcl = clinical_train['PatientID'].tolist()
    for k in range(len(IDcl)):
        identity = str(IDcl[k])
        if len(identity)==1:
            IDcl[k]='00'+identity
        elif len(identity)==2:
            IDcl[k]='0'+identity
        else:
            IDcl[k]=identity
    
    clinical_train['PatientID']=IDcl
    clinical_train.head()
    
    
    clinical_train['Histology']=clinical_train['Histology'].fillna('inconnu')
    clinical_train['Histology']=clinical_train['Histology'].str.lower()
    clinical_train['Histology']=clinical_train['Histology'].replace(['nsclc nos (not otherwise specified)'],'nos')
    temp=pd.get_dummies(clinical_train['Histology'])
    temp.index = clinical_train.index
    temp = temp.astype('int64')
    temp.dtypes
    clinical_train = pd.merge(clinical_train,temp,left_index=True, right_index=True)
    clinical_train.pop('Histology')
    
    
    moy = clinical_train['age'].mean(skipna=True)
    clinical_train['age']=clinical_train['age'].fillna(moy)
    
    
    sansID = radiomic_train[list(radiomic_train.columns)[1:]]
    
    sansID = sansID.astype(float)
    
    sansID.describe()
    
    files = listdir('/home/herearii/Documents/Owkin/x_test/images')
    
    name_test = []
    scan_test = []
    mask_test = []
    for file in files:
        path = '/home/herearii/Documents/Owkin/x_test/images/'+file
        archive_test = np.load(path)
        name_test.append(file[8:11])
        scan_test.append(archive_test['scan'])
        mask_test.append(archive_test['mask'])
    image_test = pd.DataFrame({'PatientID':name_test,'CT scan':scan_test,'mask':mask_test})
    
    
    radiomic_test = pd.read_csv('/home/herearii/Documents/Owkin/x_test/features/radiomics.csv')
    
    clinical_test = pd.read_csv('/home/herearii/Documents/Owkin/x_test/features/clinical_data.csv')
    
    
    IDcl = clinical_test['PatientID'].tolist()
    for k in range(len(IDcl)):
        identity = str(IDcl[k])
        if len(identity)==1:
            IDcl[k]='00'+identity
        elif len(identity)==2:
            IDcl[k]='0'+identity
        else:
            IDcl[k]=identity
    
    clinical_test['PatientID']=IDcl
    clinical_test.head()
    
    
    clinical_test['Histology']=clinical_test['Histology'].fillna('inconnu')
    clinical_test['Histology']=clinical_test['Histology'].str.lower()
    clinical_test['Histology']=clinical_test['Histology'].replace(['nsclc nos (not otherwise specified)'],'nos')
    temp=pd.get_dummies(clinical_test['Histology'])
    temp.index = clinical_test.index
    temp = temp.astype('int64')
    clinical_test = pd.merge(clinical_test,temp,left_index=True, right_index=True)
    clinical_test.pop('Histology')
    moy = clinical_test['age'].mean(skipna=True)
    clinical_test['age']=clinical_test['age'].fillna(moy)
    
    
    r0=radiomic_test.iloc[[0]].values.tolist()
    r1=radiomic_test.iloc[[1]].values.tolist()
    
    r0[0][0]=r1[0][0]
    
    
    radiomic_test = radiomic_test.set_axis(r0[0],axis=1)
    
    radiomic_test = radiomic_test.drop([0,1])
    
    radiomic_test.head()
    
    
    
    radiomic_train
    radiomic_train.index=radiomic_train['PatientID']
    radiomic_train.pop('PatientID')
    radiomic_train
    
    
    radiomic_test.index=radiomic_test['PatientID']
    radiomic_test.pop('PatientID')
    
    
    radiomic_test = radiomic_test.astype('float')
    radiomic_train = radiomic_train.astype('float')
    
    
    clinical_train.index=clinical_train['PatientID']
    clinical_train.pop('PatientID')
    
    clinical_test.index=clinical_test['PatientID']
    clinical_test.pop('PatientID')
    
    image_test.index=image_test['PatientID']
    image_test.pop('PatientID')
    
    
    x_train = pd.merge(radiomic_train,clinical_train,left_index=True, right_index=True)
    x_test = pd.merge(radiomic_test,clinical_test,left_index=True, right_index=True)
    
    
    y_train = pd.read_csv('/home/herearii/Documents/Owkin/y_train.csv')
    y_train.index = y_train['PatientID']
    y_train.pop('PatientID')
    
    
    x_train.pop('SourceDataset')
    
    x_test.pop('SourceDataset')
    return x_train,x_test,y_train
    
#x_train,x_test,y_train = deuxieme()



from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

#Test VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

x_tr.index = x_tr_clean.index.astype(int)
y_tr.index = y_tr.index.astype(int)
newdata = pd.merge(x_tr,y_tr, left_index=True, right_index=True)

x_tr_numeric_data = newdata.drop(columns=['Mstage', 'Nstage',
'Tstage', 'adenocarcinoma', 'inconnu', 'large cell', 'nos',
'squamous cell carcinoma', 'SurvivalTime','Event'])

def vif_test(newdata):

    X = newdata
    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info['Column'] = X.columns
    vif_info.sort_values('VIF', ascending=False)
    print(vif_info[vif_info['VIF']<10])
vif_test(x_tr_numeric_data)
x_tr_clean=newdata[['Mstage','Nstage','adenocarcinoma','inconnu','large cell']]

#Correlation
import seaborn as sns
sns.heatmap(newdata.corr())

#Cox PH regression
from lifelines import CoxPHFitter
#cph = CoxPHFitter(event_col='Event',l1_ratio=0.01432,n_alphas=155)

x_tr_clean.index = x_tr_clean.index.astype(int)
y_tr.index = y_tr.index.astype(int)
data = pd.merge(x_tr_clean,y_tr , left_index=True, right_index=True)

cph = CoxPHFitter()
cph.fit(data, duration_col='SurvivalTime',event_col='Event')

#CV method 1

from lifelines.utils.sklearn_adapter import sklearn_adapter
from lifelines import CoxPHFitter

CoxRegression = sklearn_adapter(CoxPHFitter,event_col='Event')

sk_cph = CoxRegression(penalizer=1e-5)
#y_tr.index = y_tr.index.astype(str)
x_tr_clean.index = x_tr_clean.index.astype(int)
y_tr.index = y_tr.index.astype(int)
data_x = pd.merge(x_tr_clean,y_tr['Event'] , left_index=True, right_index=True)
data_y = y_tr['SurvivalTime']
sk_cph.fit(data_x, data_y) 
print(sk_cph)

##Vérification
pl = sk_cph.predict(x_ts) #0.6325874125856431
y_test = pd.DataFrame(np.array([[y,np.nan] for y in pl]),index=x_ts.index)
y_test.columns = ['SurvivalTime','Event']
y_train.index=y_train.index.astype('int64')
y_test.index=y_test.index.astype('int64')
print(cindex(y_ts, y_test)) 

#CV method 2
from lifelines import WeibullAFTFitter
from sklearn.model_selection import cross_val_score


base_class = sklearn_adapter(WeibullAFTFitter, event_col='Event')
wf = base_class()
x_tr_clean.index = x_tr_clean.index.astype(int)
y_tr.index = y_tr.index.astype(int)
data_x = pd.merge(x_tr_clean,y_tr['Event'] , left_index=True, right_index=True)
data_y = y_tr['SurvivalTime']

scores = cross_val_score(wf, data_x, data_y, cv=5)
print(scores)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

ccindex=make_scorer(cindex)
clf = GridSearchCV(wf, {
   "penalizer": 10.0 ** np.arange(-2, 3),
   "l1_ratio": [0, 1/3, 2/3],
   "model_ancillary": [True, False],
}, scoring=ccindex,cv=5)
clf.fit(data_x, data_y)

print(clf.best_estimator_)

model = clf.best_estimator_

pl = model.predict(x_ts)
y_test = pd.DataFrame(np.array([[y,np.nan] for y in pl]),index=x_ts.index)
y_test.columns = ['SurvivalTime','Event']
y_train.index=y_train.index.astype('int64')
y_test.index=y_test.index.astype('int64')
print(cindex(y_ts, y_test)) 

#CV
from sklearn.model_selection import GridSearchCV

param_grid = {}
grid = GridSearchCV(CoxPHFitter(),param_grid,refit=True,verbose = 3, scoring=)
grid.fit(data)

from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.metrics import make_scorer
ccindex=make_scorer(cindex)

cph = CoxPHFitter()
scores = k_fold_cross_validation(cph, data, duration_col='SurvivalTime',event_col='Event', k=3)
print(scores)
# [-1.7156375401462247, -1.7559226410789432, -1.8146573330281217]

scores = k_fold_cross_validation(cph, data, duration_col='SurvivalTime',event_col='Event', k=3, scoring_method="concordance_index")
print(scores)
# [0.4938347718865598, 0.6709211986681465, 0.5893958076448829]

#Prediction
y_test = cph.predict_survival_function(x_ts)
y_pred = []
for pred in y_test:
    time = y_test.index
    prob = list(y_test[pred])
    i = 0
    while i < len(prob)-1 and prob[i] > 0.9:
        i+=1
    y_pred.append(time[i])

#Vérification

y_test = pd.DataFrame(np.array([[y,np.nan] for y in y_pred]),index=x_ts.index) #0.6332867132849419 Model Cox sans CV
y_test.columns = ['SurvivalTime','Event']
y_train.index=y_train.index.astype('int64')
y_test.index=y_test.index.astype('int64')
print(cindex(y_ts, y_test))


#Export
y_test = cph.predict_survival_function(x_test)
y_pred = []
for pred in y_test:
    time = y_test.index
    prob = list(y_test[pred])
    i = 0
    while i < len(prob)-1 and prob[i] > 0.9:
        i+=1
    y_pred.append(time[i])

y_test = pd.DataFrame(np.array([[y,np.nan] for y in y_pred]),index=x_test.index)
y_test.columns = ['SurvivalTime','Event']
y_train.index=y_train.index.astype('int64')
y_test.index=y_test.index.astype('int64')
y_test.to_csv('/home/herearii/Documents/Owkin/owkin_coxph.csv')

