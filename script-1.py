import numpy as np
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import scipy as sp
from sklearn.linear_model import LinearRegression
import autosklearn.regression
from lifelines import CoxPHFitter

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

## Preprocessing (2)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_train_pca = pd.DataFrame(x_train_pca)
x_train_pca.index = x_train.index
x_train_pca.columns = x_train.columns

pca.fit(x_test)
x_test_pca = pca.transform(x_test)
x_test_pca = pd.DataFrame(x_test_pca)
x_test_pca.index = x_test.index
x_test_pca.columns = x_test.columns
#Feature Selection

## Auto ML

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30
    )

automl.fit(x_train,y_train['SurvivalTime'])
y_test = automl.predict(x_train)
y_test = pd.DataFrame(np.array([[y,np.nan] for y in y_test]),index=x_train.index)
y_test.columns = ['SurvivalTime','Event']
y_test.index=y_test.index.astype('int64')
cindex(y_train, y_test) #0.6825

y_test = automl.predict(x_test)
y_test = pd.DataFrame(np.array([[y,np.nan] for y in y_test]),index=x_test.index)
y_test.columns = ['SurvivalTime','Event']
y_test.to_csv('/home/herearii/Documents/Owkin/owkin_autoML.csv')

## Manual ################################################
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis

data_x_numeric = OneHotEncoder().fit_transform(x_train)
y_train['Event'] = [str(k) for k in y_train['Event']]
y_train = pd.DataFrame({'Event':y_train['Event'],'SurvivalTime':y_train['SurvivalTime']})
estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, y_train['SurvivalTime'])

#RandomForest
from sklearn.ensemble import RandomForestClassifier

predictor = RandomForestClassifier()
x = predictor.fit(data_x_numeric, y_train['SurvivalTime'])
