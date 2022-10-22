import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.base import TransformerMixin


# Impute des häufigsten Wertes:
class Imputer_Most_Frequent(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, orig_df, y=None):
        for c in orig_df.city.unique():
            imp_df = pd.DataFrame(columns=orig_df.columns)
            imp = SimpleImputer(strategy="most_frequent")
            np_arr = imp.fit_transform(orig_df[orig_df['city']==c])
            _df = pd.DataFrame(np_arr, columns = orig_df.columns)
            imp_df = pd.concat([imp_df, _df])

        imp_df.reset_index(inplace=True)
        imp_df = imp_df.astype(orig_df.dtypes.to_dict()).drop('index', axis=1)
        return imp_df

Imputer_Most_Frequent


orig_df = pd.read_csv('../data/dengue_features_train.csv')
labels_df =  pd.read_csv('../data/dengue_labels_train.csv')

orig_df = pd.concat([orig_df,labels_df['total_cases']], axis=1)

imputer = SimpleImputer(strategy="most_frequent")

catboost = CatBoostRegressor()

imputer2 = Imputer_Most_Frequent()



pipe = Pipeline([('impute',imputer2),('catboost',catboost)])

orig_df['city'] = orig_df['city'].map({'sj':0, 'iq':1})
y = orig_df['total_cases']
X = orig_df.drop(['total_cases', 'week_start_date'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)

print(pipe.score(X, y))
print(mean_absolute_error(y_test, pipe.predict(X_test)))
print(mean_absolute_error(np.exp(y_test), np.exp(pipe.predict(X_test))))





#
# reg = CatBoostRegressor().fit(X_train, y_train)
# print(reg.score(X, y))
# print(mean_absolute_error(y_test, reg.predict(X_test)))
# print(mean_absolute_error(np.exp(y_test), np.exp(reg.predict(X_test))))
#
#
#
#
# def impute_by_city(orig_df):
#
#     for c in orig_df.city.unique():
#         imp_df = pd.DataFrame(columns=orig_df.columns)
#         imp = SimpleImputer(strategy="most_frequent")
#         np_arr = imp.fit_transform(orig_df[orig_df['city']==c])
#         _df = pd.DataFrame(np_arr, columns = orig_df.columns)
#         imp_df = pd.concat([imp_df, _df])
#
#     imp_df.reset_index(inplace=True)
#     imp_df = imp_df.astype(orig_df.dtypes.to_dict()).drop('index', axis=1)
#
#     df = imp_df
#     df = pd.concat([df,labels_df['total_cases']], axis=1)
#
#     return df
#
# df['city'] = df['city'].map({'sj':0, 'iq':1})
#
#
# y = df['total_cases']
# X = df.drop(['total_cases', 'week_start_date'],axis=1)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
#
#
#
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# # The pipeline can be used as any other estimator
# # and avoids leaking the test set into the train set
# pipe.fit(X_train, y_train)
#
# pipe.score(X_test, y_test)