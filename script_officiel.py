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

from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

grade_str = y_train['Event'].astype('object').values[:,np.newaxis]
grade_num = OrdinalEncoder(categories=[[0,1]]).fit_transform(grade_str)
Xt = OneHotEncoder().fit_transform(x_train)
#Xt = np.column_stack((Xt.values,grade_num))

newy = []
for k in y_train['Event']:
    if k==1:
        newy.append(True)
    else:
        newy.append(False)

y1 = list(zip(newy,y_train['SurvivalTime']))

y2 = np.array(y1,dtype=[('cens', '?'), ('time', '<f8')])

X_train1, X_test1, Y_train1,Y_test1 = train_test_split(Xt.values,y2,
                                                       test_size=0.25,
                                                       random_state=20)
rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features='sqrt',
                           n_jobs=-1,
                           random_state=20)

rsf.fit(Xt,y2)
#rsf.score(X_test1,Y_test1) #0.806776

grade_str = y_train['Event'].astype('object').values[:,np.newaxis]
grade_num = OrdinalEncoder(categories=[[0,1]]).fit_transform(grade_str)
Xt1 = OneHotEncoder().fit_transform(x_test)
Xt1 = Xt1.values


m=pd.Series(rsf.predict(Xt))

v = pd.DataFrame(np.array([[k,np.nan] for k in list(m)]),index=y_train.index)
v.columns = ['SurvivalTime','Event']
v.to_csv('/home/herearii/Documents/Owkin/owkin_random_survival_forest.csv')

from metrics_t9gbvr2 import cindex

cindex.concord_index(y_train, v)
#0.224

#Pensé à faire la validation croisé

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':[200,500],
    'max_features':['auto','sqrt','log2'],
    'max_depth':[4,5,6,7,8,9,10,11,12,13,14,15],
    'criterion':['gini','entropy']
    }

GridS = GridSearchCV(estimator=rsf,param_grid=param_grid,cv=5)

GridS.fit(Xt,y2)

#CoxPHSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

def selection_feature(X,y):
    n = X.shape[1]
    scores = np.empty(n)
    m = CoxPHSurvivalAnalysis()
    for j in range(n):
        Xjj = X[:,j:j+1]
        m.fit(Xjj,y)
        scores[j] = m.score(Xjj,y)
    return scores

relevant_feat = selection_feature(Xt.values,y2)
feat = pd.Series(relevant_feat,index=Xt.columns).sort_values(ascending=False)

       
essai_2 = Xt[rel_feat]

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

pipe = Pipeline([('encode',OneHotEncoder()),
                 ('select',SelectKBest(selection_feature,k=3)),
                 ('model',CoxPHSurvivalAnalysis())])

from sklearn.model_selection import GridSearchCV, KFold

param_grid = {'select__k' : np.arange(1,Xt.shape[1]+1) }
cv = KFold(n_splits=3,random_state=1,shuffle=True)
gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
gcv.fit(Xt,y2)

results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score',ascending=False)
results.loc[:,~ results.columns.str.endswith("_time")]

pipe.set_params(**gcv.best_params_)
pipe.fit(Xt,y2)

encoder,transformer,final_estimator = [s[1] for s in pipe.steps]
vv = pd.Series(final_estimator.coef_,index=encoder.encoded_columns_[transformer.get_support()])

bes_feat = Xt[list(vv.index)]
rsf.fit(bes_feat,y2)

m=pd.Series(rsf.predict(bes_feat))
v = pd.DataFrame(np.array([[k,np.nan] for k in list(m)]),index=y_train.index)
v.columns = ['SurvivalTime','Event']

from metrics_t9gbvr2 import cindex
cindex.concord_index(y_train, v)

grade_str = y_train['Event'].astype('object').values[:,np.newaxis]
grade_num = OrdinalEncoder(categories=[[0,1]]).fit_transform(grade_str)
Xt1 = OneHotEncoder().fit_transform(x_test)
Xt1 = Xt1.values
bes_feat = Xt1[list(vv.index)]

m=pd.Series(rsf.predict(bes_feat))
v = pd.DataFrame(np.array([[k,np.nan] for k in list(m)]),index=x_test.index)
v.columns = ['SurvivalTime','Event']
v.to_csv('/home/herearii/Documents/Owkin/owkin_random_survival_forest.csv')
