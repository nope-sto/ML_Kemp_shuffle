import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import model_selection, metrics
import joblib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import random
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

class featureselector:
    def __init__(self):
        self.not_numeric=[]      
    def prepare_data(self,X,y):
        X = pd.read_csv(X, sep=';')
        X = X.iloc[: , 1:]
        y = pd.read_csv(y, sep=';')
        return(X,y)
    def select_features(self,X_full,y,number_features):
        clf = xgb.XGBRegressor()#,n_estimators=200,colsample_bytree= 0.8945064111389341, gamma=  0.09061653308283008, learning_rate= 0.04324813368338875, max_depth= 6, subsample=  0.8157405806739012,min_child_weight=1)
        model = clf.fit(X_full, y)
        importance = model.feature_importances_
        df_importance = pd.DataFrame([importance])
        df_importance = pd.DataFrame(data=df_importance.values,columns=X_full.columns).sort_values(0, axis=1, ascending=False)
        Top_X  = df_importance.iloc[: , :number_features]
        selected_features = list(Top_X.columns)
        df_sel_features = pd.DataFrame([selected_features])
        df_sel_features.to_csv('Top_{}_selected_features.tsv'.format(number_features), sep="\t")
        #crop dataframe according to selected features
        X_selected = X_full.loc[:, selected_features]
        print("Selected Top {} Features:".format(number_features))
        return(selected_features,X_selected)
    def train(self,X,y,final_model):#X dependent on number of selected features
        clf = xgb.XGBRegressor()
        model=clf.fit(X, y)
        #cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        rmse = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
        #print(results)
        print("RMSD: {} +/- {}".format(rmse.mean(), rmse.std()))         
        r2 = cross_val_score(model, X, y, scoring='r2', cv=cv)
        print("R2: {} +/- {}".format(r2.mean(), r2.std()))
        return([r2.mean(),r2.std(),rmse.mean(), rmse.std()])
featureselector=featureselector()        
