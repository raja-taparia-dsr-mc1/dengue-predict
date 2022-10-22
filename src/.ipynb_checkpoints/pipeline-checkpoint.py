# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:32:49 2022

@author: hannes
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error

def pipeline(X: pd.DataFrame, y: pd.DataFrame, categorical_vars: list):

    preprocessor = ColumnTransformer(
        transformers=[
            ('missing_imputation', SimpleImputer(strategy='mean'), X.drop(columns=['city','week_start_date','weekofyear']).columns),
            ('encodings', OneHotEncoder(), categorical_vars)
            ])

    estimator = RandomForestRegressor(random_state=42, n_estimators=100)

    pipe = Pipeline([('preprocessing', preprocessor),('estimator', estimator)]) 

    preds = cross_val_predict(pipe, X, y)
    
    mae = mean_absolute_error(y,preds)
    
    return mae