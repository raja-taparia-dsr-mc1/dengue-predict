# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:20:12 2022

@author: hannes
"""

import numpy as np
import pandas as pd
import math

def feature_engineering(data, train=True, city='San Juan'):
    
    data.fillna(method='ffill', inplace=True)
        
    if city == 'San Juan':
        data = data[data.city=='sj']
       
        data["cos_weekofyear"] = data["weekofyear"].apply(lambda x: np.cos(2 * math.pi * (x-1) / 52))
            
    else:
        data = data[data.city=='iq']

        for c, l in zip(['reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'reanalysis_precip_amt_kg_per_m2'], [4,4,4]):
            data[c+'_rolling'+str(l)] = data[c].rolling(l).mean()
            
   
        data['month'] = pd.to_datetime(data.week_start_date).dt.month
        
    if train:
        X = data.drop(columns='total_cases')
        y = data['total_cases']
        return X, y

    else:
        return data