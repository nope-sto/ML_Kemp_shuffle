import pandas as pd 
from pandas import MultiIndex, Int16Dtype
import numpy as np
import sklearn
import xgboost as xgb
from featureselector_main import featureselector

#path to features + real Tms (.csv file) for training
X_data = 'X_clean.csv'
y_data = 'y.csv'
#path to model
final_model="prelim_model_kemp.json"
#evaluation
eval_=pd.DataFrame([['r2','r2_stdv','rmse','rmse_stdv']])
eval_.to_csv('feature_selection_monitor_kemp.csv',sep=',', header=False, index=False)
#number of features for feature selection
for number_features in range(1,100):
    #print('preparing data (60 seconds)...')
    X,y=featureselector.prepare_data(X_data,y_data)
    #print('selecting features...')
    selected_features,X_selected=featureselector.select_features(X,y,number_features)
    #print('training model...')
    r2_rmse = featureselector.train(X_selected,y,final_model)
    eval_df = pd.DataFrame([r2_rmse])
    eval_df.to_csv('feature_selection_monitor_kemp.csv', mode='a',sep=',', header=False, index=False)
