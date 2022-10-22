# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:11:18 2022

@author: hanne
"""

import pandas as pd
import numpy as np
from pipeline_hannes import pipeline
from pipeline_for_submission import pipeline_for_submission
import math

data_path = '/home/jan/Documents/Study/DSR/dengue-predict/data/'

train = pd.read_csv(data_path+'dengue_features_train.csv')

test = pd.read_csv(data_path+'dengue_features_test.csv')

train_labels = pd.read_csv(data_path+'dengue_labels_train.csv')

train = train.merge(train_labels, on=['city','year','weekofyear'], how='left')

train["cos_weekofyear"] = train["weekofyear"].apply(lambda x: np.cos(2 * math.pi * (x-1) / 52))
test["cos_weekofyear"] = test["weekofyear"].apply(lambda x: np.cos(2 * math.pi * (x-1) / 52))

#train['total_cases'] = np.log(train['total_cases']+1)


submission_df = pd.read_csv(data_path+'submission_format.csv')

train_sj = train[train.city=='sj']
train_iq = train[train.city=='iq']

test_sj = test[test.city=='sj']
test_iq = test[test.city=='iq']

train_sj.fillna(method='ffill', inplace=True)
train_iq.fillna(method='ffill', inplace=True)

#for c, l in zip(['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'reanalysis_precip_amt_kg_per_m2'], [14,15,9]):
#    train_sj[c+'_rolling'+str(l)] = train_sj[c].rolling(l).mean()

#for c, l in zip(['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'reanalysis_precip_amt_kg_per_m2'], [4,4,3]):
#    train_iq[c+'_rolling'+str(l)] = train_iq[c].rolling(l).mean()
    
X_sj = train_sj.drop(columns='total_cases')
y_sj = train_sj['total_cases']

X_iq = train_iq.drop(columns='total_cases')
y_iq = train_iq['total_cases']

#mae, predictions_sj = pipeline(X_sj, y_sj, test_sj, categorical_vars=['weekofyear'])
#mae, predictions_sj = pipeline(X_iq, y_iq, test_iq, categorical_vars=['weekofyear'])

predictions_sj = pipeline_for_submission(X_sj, y_sj, test_sj, categorical_vars=['weekofyear'])

predictions_iq = pipeline_for_submission(X_iq, y_iq, test_iq, categorical_vars=['weekofyear'])

predictions = np.concatenate((predictions_sj, predictions_iq))

submission_df['total_cases'] = predictions

submission_df['total_cases'] = submission_df['total_cases'].apply(lambda x: np.int(np.round(x,0)))

submission_df.to_csv(data_path+'submission_df.csv', index=False)

print(submission_df)
print(mae)