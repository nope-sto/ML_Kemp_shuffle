import pandas as pd 
from pandas import MultiIndex, Int16Dtype
import numpy as np
import sklearn
import xgboost as xgb
from prediction_main import PRED

#path to trained model
final_model='model_80_tuned.pmml'

#path to features + real Tms (.csv file) for training
X = pd.read_csv("seq_features.csv", delimiter=";")
X_id=X["Tm"]
X_id.to_csv("IDS.csv")
