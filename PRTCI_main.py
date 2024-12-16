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
import pickle

class PRTCI:
    def __init__(self):
        self.not_numeric=[]      
    def prepare_data(self,X,y):
        X = pd.read_csv(X, sep=';')      
        y = pd.read_csv(y, sep=';')
        return(X,y)
    def select_features(self,X_full,y,number_features):
        #X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_full, y, test_size=0.3,random_state=0)
        clf = xgb.XGBRegressor()
        model = clf.fit(X_full, y)
        importance = model.feature_importances_
        df_importance = pd.DataFrame([importance])
        df_importance = pd.DataFrame(data=df_importance.values,columns=X_full.columns).sort_values(0, axis=1, ascending=False)
        Top_X  = df_importance.iloc[: , :number_features]
        selected_features = list(Top_X.columns)
        hbars = plt.barh(list(df_importance.iloc[: , 1:number_features]), importance[1:number_features])
        plt.xlabel("Xgboost Feature Importance")
        ##plt.xscale(value='log')
        plt.autoscale()
        plt.bar_label(hbars)
        #plt.xticks(rotation=90)
        plt.savefig('Feature_Importance.png')
        #importance = permutation_importance(model, X_test, Y_test).argsort()
        df_sel_features = pd.DataFrame([selected_features])
        df_sel_features.to_csv('Top_{}_selected_features.tsv'.format(number_features), sep="\t")
        #crop dataframe according to selected features
        X_selected = X_full.loc[:, selected_features]
        print("Selected Top {} Features:".format(number_features))
        return(selected_features,X_selected)
    def train(self,X,y,final_model):#X dependent on number of selected features
        #clf=xgb.XGBRegressor(tree_method="gpu_hist", sampling_method="gradient_based", booster="gbtree",n_estimators=200,colsample_bytree= 0.5281391785761806, gamma=  0.3602580306957324, learning_rate= 0.04117068125562073, max_depth= 9, subsample=   0.718448954862998,min_child_weight=4) 
        clf = xgb.XGBRegressor()
        model=clf.fit(X, y)
        pickle.dump(clf, open(final_model, "wb"))
        ###cross validation
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        results = cross_val_score(clf, X, y, scoring='neg_root_mean_squared_error', cv=cv)
        print("RMSD: {} +/- {}".format(results.mean(), results.std()))         
        results = cross_val_score(clf, X, y, scoring='r2', cv=cv)
        print("R2: {} +/- {}".format(results.mean(), results.std()))
        print('Successful Training! Final model is saved as {}'.format(final_model))
    def apply(self,final_model,X_full,prediction):
        from pypmml import Model
        model = Model.fromFile(final_model)
        y_pred = pd.DataFrame(model.predict(X_full))
        y_pred = y_pred.rename(columns={0: 'Tm_pred'})
        y_pred.to_csv(prediction, sep="\t")
        print('Successful Prediction!  {}'.format(prediction))
PRTCI=PRTCI()        
