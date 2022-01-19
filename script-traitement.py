#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:00:42 2022

@author: herearii
"""

#Traitement de la multicolinéarité

x_tr_data = pd.merge(x_train,y_train.sort_index(), left_index=True, right_index=True,sort=True,how='inner')
x_tr_data_qt = x_tr_data.drop(columns=['SourceDataset','Histology_large cell', 'Histology_nos',
'Histology_squamous cell carcinoma', 'Tstage_2', 'Tstage_3', 'Tstage_4',
'Nstage_1', 'Nstage_2', 'Nstage_3', 'SurvivalTime', 'Event'])

##Correlation
import seaborn as sns
sns.heatmap(x_tr_data_qt.corr())

strongly_corr = ["original_shape_Compactness1", "original_shape_Compactness2", 
                                             "original_shape_SphericalDisproportion", "original_shape_Sphericity", 
                                             "original_firstorder_Energy", "original_firstorder_Minimum", 
                                             "original_glcm_ClusterProminence", "original_glrlm_LongRunLowGrayLevelEmphasis"]

x_tr_data_qt_=x_tr_data_qt.drop(columns=strongly_corr)
##VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_test(newdata):
    X = newdata
    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info['Column'] = X.columns
    vif_info.sort_values('VIF', ascending=False)
    return vif_info[vif_info['VIF']<=10]
print(vif_test(x_tr_data_qt_))

to_remove = ["original_glcm_Id","original_glcm_Idm","original_firstorder_Range", 
                                               "original_glcm_JointEntropy", "original_glrlm_HighGrayLevelRunEmphasis",
                                               "original_firstorder_Entropy","original_firstorder_Uniformity", 
                                               "original_firstorder_Variance","original_glcm_Autocorrelation",
                                               "original_glcm_DifferenceAverage","original_glcm_DifferenceEntropy",
                                               "original_glcm_Idn","original_firstorder_MeanAbsoluteDeviation",
                                               "original_firstorder_RootMeanSquared","original_glrlm_RunPercentage",
                                               "original_glrlm_ShortRunLowGrayLevelEmphasis","original_glcm_SumEntropy",
                                               "original_glcm_SumAverage","original_firstorder_StandardDeviation",
                                               "original_firstorder_Mean","original_glcm_JointEnergy","original_shape_VoxelVolume",
                                               "original_glrlm_ShortRunEmphasis", "original_glrlm_LongRunHighGrayLevelEmphasis",
                                               "original_shape_SurfaceArea","original_glcm_Contrast","original_glcm_Imc1",
                                               "original_glrlm_LongRunEmphasis"]
x_tr_clean=x_tr_data_qt_.drop(columns=to_remove)

variable_to_remove = to_remove+strongly_corr

#Traitement
from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
x_tr_clean=x_tr.drop(columns=variable_to_remove)
x_ts_clean=x_ts.drop(columns=variable_to_remove)

#Cox PH regression
from lifelines import CoxPHFitter
#cph = CoxPHFitter(event_col='Event',l1_ratio=0.01432,n_alphas=155)

x_tr_ech = pd.merge(x_tr,y_tr, left_index=True, right_index=True,sort=True,how='inner')
data_tr = x_tr_ech.drop(columns=variable_to_remove)

x_ts_ech = pd.merge(x_ts,y_ts, left_index=True, right_index=True,sort=True,how='inner')
data_ts = x_ts_ech.drop(columns=variable_to_remove)

cph = CoxPHFitter()
cph.fit(data_tr, duration_col='SurvivalTime',event_col='Event')
cph.print_summary()


x_tr_data_final = data_tr.drop(columns=["original_glcm_ClusterShade","original_glrlm_RunLengthNonUniformity",
                                                    "original_shape_Maximum3DDiameter","original_glcm_Imc2",
                                                    "original_glrlm_LowGrayLevelRunEmphasis",
                                                    "original_glrlm_ShortRunHighGrayLevelEmphasis","original_glcm_Idmn",
                                                    "original_firstorder_Kurtosis","Tstage_2","Tstage_3","Tstage_4",
                                                    "original_glcm_Correlation","original_firstorder_Skewness", 
                                                    "original_glcm_InverseVariance","original_shape_SurfaceVolumeRatio",
                                                    "original_glcm_MaximumProbability",'Histology_large cell','Histology_nos',
                                                    'Histology_squamous cell carcinoma',"original_firstorder_Maximum"])

cph.fit(x_tr_data_final, duration_col='SurvivalTime',event_col='Event')
cph.print_summary()

#C-index Train

from lifelines.utils import concordance_index
print(concordance_index(x_tr_data_final['SurvivalTime'], -cph.predict_partial_hazard(x_tr_data_final), x_tr_data_final['Event']))
#0.7236248119043638 pour le train data

x_ts_data_final = data_ts.drop(columns=["original_glcm_ClusterShade","original_glrlm_RunLengthNonUniformity",
                                                    "original_shape_Maximum3DDiameter","original_glcm_Imc2",
                                                    "original_glrlm_LowGrayLevelRunEmphasis",
                                                    "original_glrlm_ShortRunHighGrayLevelEmphasis","original_glcm_Idmn",
                                                    "original_firstorder_Kurtosis","Tstage_2","Tstage_3","Tstage_4",
                                                    "original_glcm_Correlation","original_firstorder_Skewness", 
                                                    "original_glcm_InverseVariance","original_shape_SurfaceVolumeRatio",
                                                    "original_glcm_MaximumProbability",'Histology_large cell','Histology_nos',
                                                    'Histology_squamous cell carcinoma',"original_firstorder_Maximum"])
#C-index Test
print(concordance_index(x_ts_data_final['SurvivalTime'], -cph.predict_partial_hazard(x_ts_data_final), x_ts_data_final['Event']))
#0.6161137440758294

#Exporation

x_test_model = x_test[['original_firstorder_Median', 'original_glcm_ClusterTendency',
       'original_glrlm_GrayLevelNonUniformity', 'age', 'SourceDataset',
       'Nstage_1', 'Nstage_2', 'Nstage_3']]
prediction_test_cph=cph.predict_expectation(x_test_model)

y_test = pd.DataFrame(np.array([[y,np.nan] for y in prediction_test_cph]),index=x_test.index)
y_test.columns = ['SurvivalTime','Event']
y_train.index=y_train.index.astype('int64')
y_test.index=y_test.index.astype('int64')
y_test.to_csv('/home/herearii/Documents/Owkin/owkin_coxph_prediction.csv')
