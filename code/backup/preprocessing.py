# Author: Jiang Hao <nju.jianghao@foxmail.com>

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.feature_extraction.text import TfidfTransformer

# read data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# combine together
df_test['TripType'] = 'UNKNOWN'
df_all = pd.concat((df_train, df_test))
df_all.fillna({'Upc': 0, 'DepartmentDescription': 'UNKNOWN', 'FinelineNumber': 10000}, inplace=True)

# encode weekdays
le_weekday = LabelEncoder()
df_all['Weekday'] = le_weekday.fit_transform(df_all['Weekday'])

# transform department desc to dummies
df_dummy = pd.get_dummies(df_all['DepartmentDescription'], prefix='buy')
df_all = pd.concat((df_all, df_dummy), axis=1)
df_dummy = pd.get_dummies(df_all['DepartmentDescription'], prefix='return', sparse=True)
df_all = pd.concat((df_all, df_dummy), axis=1)
df_dummy = pd.get_dummies(df_all['FinelineNumber'], prefix='fineline', sparse=True)
df_all = pd.concat((df_all, df_dummy), axis=1)
columns_buy = [column for column in df_all.columns if 'buy_' in column]
columns_return = [column for column in df_all.columns if 'return_' in column]
columns_fineline = [column for column in df_all.columns if 'fineline_' in column]

df_all['BuyNum'] = df_all['ScanCount'].apply(lambda x: x if x>0 else 0)
df_all['ReturnNum'] = df_all['ScanCount'].apply(lambda x: -x if x<0 else 0) # pls note that return_num has been changed to positive

# make dummy
df_all[columns_buy] = df_all[columns_buy].apply(lambda x: np.asarray(x) * np.asarray(df_all['BuyNum']))
df_all[columns_return] = df_all[columns_return].apply(lambda x: np.asarray(x) * np.asarray(df_all['ReturnNum']))
df_all[columns_fineline] = df_all[columns_fineline].apply(lambda x: np.asarray(x) * np.asarray(df_all['BuyNum']))

df_all.drop(['Upc', 'DepartmentDescription', 'FinelineNumber', 'ScanCount'], axis=1, inplace=True)

# groupby and generate features
df_grouped = df_all.groupby('VisitNumber', sort=False)
df_transformed = df_grouped[['Weekday', 'TripType']].agg(lambda x: x.iloc[0]).reset_index()
df_transformed[columns_buy + columns_return + columns_fineline] = df_grouped[columns_buy + columns_return + columns_fineline].agg(np.sum).reset_index(drop=True)
df_transformed['HasReturn'] = df_grouped['ReturnNum'].agg(lambda x: 1 if (np.sum(x) != 0) else 0)
df_transformed['OnlyReturn'] = (df_grouped['BuyNum'].agg(np.sum) == 0) & (df_grouped['ReturnNum'].agg(np.sum) != 0)
df_transformed['OnlyReturn'] = df_transformed['OnlyReturn'].apply(lambda x: 1 if x is True else 0)

# # transform using tfidf
# tfidf = TfidfTransformer()
# df_transformed[columns_buy + columns_return] = tfidf.fit_transform(df_transformed[columns_buy + columns_return]).A

df_train = df_transformed[df_transformed['TripType'] != 'UNKNOWN']
df_test = df_transformed[df_transformed['TripType'] == 'UNKNOWN']

# x_train = np.array(df_train.drop(['VisitNumber', 'TripType'], axis=1), np.int32)
# y_train = df_train['TripType']
# le_triptype = LabelEncoder()
# y_train = le_triptype.fit_transform(y_train)
# x_test = np.array(df_test.drop(['VisitNumber', 'TripType'], axis=1), np.int32)

# with open('./data/pack.pkl', 'wb') as f:
#     pickle.dump((x_train, y_train, x_test), f)
