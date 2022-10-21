# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:32:49 2022

@author: hannes
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

def pipeline(X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, categorical_vars: list):

    preprocessor = ColumnTransformer(
        transformers=[
            ('missing_imputation', SimpleImputer(strategy='mean'), X.drop(columns=['city','week_start_date','weekofyear']).columns),
            ('encodings', OneHotEncoder(), categorical_vars)
            ])

    estimator = lgb.LGBMRegressor(learning_rate=0.05,n_estimators=100,max_depth=5,num_leaves=10, random_state=42,min_data_in_leaf=5)
    
    pipe = Pipeline([('preprocessing', preprocessor),('estimator', estimator)]) 
    
    preds = cross_val_predict(pipe, X, y)
    mae = mean_absolute_error(y,preds)
    
    results = cross_validate(pipe, X, y, return_estimator=True)
    predictions = []
    for est in results['estimator']:
        preds_test = est.transform(X_test)
        predictions.append(preds_test)
    
    return mae, predictions