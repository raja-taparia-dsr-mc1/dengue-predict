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




        endemic_dict={'station_max_temp_c': 32.8, 'station_avg_temp_c': 28.178571, 'reanalysis_specific_humidity_g_per_kg': 17.87, 'reanalysis_sat_precip_amt_mm': 51.405, 'reanalysis_relative_humidity_percent':80.895, 'reanalysis_precip_amt_kg_per_m2': 36.36, 'reanalysis_max_air_temp_k': 302.4, 'precipitation_amt_mm': 38.44}
        for e, v in endemic_dict.items():
            data.loc[:, e+'_endemic'] =  data[e]>v
            data.loc[:, e+'_endemic'] = data.loc[:, e+'_endemic'].map({True: 1, False:0})

        #data.loc[:, 'ndvi_sw_endemic'] = ( data['ndvi_sw']>0.168871) & ( data['ndvi_sw']<0.173900)
        #data.loc[:, 'ndvi_se_endemic'] = ( data['ndvi_se']>0.180967) & ( data['ndvi_se']<0.198483)
        #data.loc[:, 'weekofyear_endemic'] = ( data['weekofyear']>38) & ( data['weekofyear']<45)

        #data.loc[:, 'ndvi_sw_endemic'] = data.loc[:, 'ndvi_sw_endemic'].map({True: 1, False:0})
        #data.loc[:, 'ndvi_se_endemic'] = data.loc[:, 'ndvi_se_endemic'].map({True: 1, False:0})
        #data.loc[:, 'weekofyear_endemic'] = data.loc[:, 'weekofyear_endemic'].map({True: 1, False:0})


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