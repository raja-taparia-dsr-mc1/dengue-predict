# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:11:18 2022

@author: hanne
"""

import pandas as pd
import numpy as np
from pipeline_adv import pipeline
from feature_engineering import feature_engineering

data_path = 'C:/Users/hanne/dengue-predict/data/'

train = pd.read_csv(data_path+'dengue_features_train.csv')

test = pd.read_csv(data_path+'dengue_features_test.csv')

train_labels = pd.read_csv(data_path+'dengue_labels_train.csv')

train = train.merge(train_labels, on=['city','year','weekofyear'], how='left')

submission_df = pd.read_csv(data_path+'submission_format.csv')

X_sj, y_sj = feature_engineering(train, city='San Juan', train=True)
test_sj = feature_engineering(test, city='San Juan', train=False)

X_iq, y_iq = feature_engineering(train, city='Iquitos', train=True)
test_iq = feature_engineering(test, city='Iquitos', train=False)


mae_sj_xgb, predictions_sj = pipeline(X_sj, y_sj, test_sj, categorical_vars=['weekofyear'], learning_rate=0.05, model='xgb')

mae_iq_xgb, predictions_iq = pipeline(X_iq, y_iq, test_iq, categorical_vars=['weekofyear','month'], learning_rate=0.01, model='xgb')

predictions_xgb = np.concatenate((predictions_sj.mean(axis=0), predictions_iq.mean(axis=0)))


mae_sj_lgbm, predictions_sj = pipeline(X_sj, y_sj, test_sj, categorical_vars=['weekofyear'], learning_rate=0.05, model='lgbm')

mae_iq_lgbm, predictions_iq = pipeline(X_iq, y_iq, test_iq, categorical_vars=['weekofyear','month'], learning_rate=0.01, model='lgbm')

predictions_lgbm = np.concatenate((predictions_sj.mean(axis=0), predictions_iq.mean(axis=0)))


mae_sj_cat, predictions_sj = pipeline(X_sj, y_sj, test_sj, categorical_vars=['weekofyear'], learning_rate=0.05, model='cat')

mae_iq_cat, predictions_iq = pipeline(X_iq, y_iq, test_iq, categorical_vars=['weekofyear','month'], learning_rate=0.01, model='cat')

predictions_cat = np.concatenate((predictions_sj.mean(axis=0), predictions_iq.mean(axis=0)))

print(f"MAE for San Juan: XGB: {mae_sj_xgb}, LGBM: {mae_sj_lgbm}, CAT: {mae_sj_cat}, MAE for Iquitos: {mae_iq_xgb}, LGBM: {mae_iq_lgbm}, CAT: {mae_iq_cat}")

predictions = predictions_xgb #(predictions_xgb + predictions_lgbm + + predictions_cat) / 3

submission_df['total_cases'] = predictions

submission_df['total_cases'] = submission_df['total_cases'].apply(lambda x: np.int(np.round(x,0)))

submission_df.to_csv(data_path+'submission_df.csv', index=False)
