# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:11:18 2022

@author: hanne
"""

import pandas as pd
from .pipeline.py import pipeline

data_path = 'C:/Users/hanne/dengue-predict/data/'

train = pd.read_csv(data_path+'dengue_features_train.csv')

train_labels = pd.read_csv(data_path+'dengue_labels_train.csv')

train_sj = train[train.city=='sj']
train_iq = train[train.city=='iq']

X_sj = train_sj.drop(columns='total_cases')
y_sj = train_sj['total_cases']

X_iq = train_sj.drop(columns='total_cases')
y_iq = train_sj['total_cases']

mae = pipeline(X_sj, y_sj)