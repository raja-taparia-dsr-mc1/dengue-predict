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
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
#from sklearn.model_selection import GridSearchCV

def pipeline(X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, categorical_vars: list, learning_rate=0.01, model='lgbm'):
    
    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('missing_imputation', SimpleImputer(strategy='mean'), X.drop(columns=['city','week_start_date','weekofyear']).columns),
            ('encodings', OneHotEncoder(), categorical_vars)
            ])
    
    # model selection
    if model == 'lgbm':
        estimator = lgb.LGBMRegressor(learning_rate=learning_rate, n_estimators=100, max_depth=5, num_leaves=10, min_data_in_leaf=5, random_state=42)
    elif model == 'xgb':
        estimator = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=100, max_depth=4, num_leaves=10, min_data_in_leaf=5, random_state=42)
    else:
        estimator = cb.CatBoostRegressor(learning_rate=learning_rate, n_estimators=100, max_depth=5, min_data_in_leaf=5, random_state=42)
        
    #gridsearch = GridSearchCV(estimator, {'learning_rate': [0.01, 0.025, 0.05, 0.1], 'max_depth': [4,5,6], 'num_leaves':[5,10,15], 'min_data_in_leaf': [5,10,15]}, scoring='neg_mean_absolute_error', refit=True)
    
    pipe = Pipeline([('preprocessing', preprocessor),('estimator', estimator)]) 
    
    # cross validation
    preds = cross_val_predict(pipe, X, y)
    mae = mean_absolute_error(y,preds)
    
    results = cross_validate(pipe, X, y, return_estimator=True, cv=5)
    
    # predict test with each of the 5-fold models
    predictions = []
    for i, est in enumerate(results['estimator']):
        preds_test = est.predict(X_test)
        predictions.append(preds_test)
    
    print('Number of folds: '+str(i+1))
    predictions = np.array(predictions)
    
    return mae, predictions