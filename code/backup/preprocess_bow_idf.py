# Author: Jiang Hao <nju.jianghao@foxmail.com>

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# read data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_train.fillna({'Upc': 0, 'DepartmentDescription': 'UNKNOWN', 'FinelineNumber': 10000}, inplace=True)
df_test.fillna({'Upc': 0, 'DepartmentDescription': 'UNKNOWN', 'FinelineNumber': 10000}, inplace=True)

def concat_feat(scan_count, department_des, fineline_num):
    department_des = '_'.join(department_des.split())
    list_feat = []
    if scan_count > 0:
        list_feat += [department_des] * scan_count
        list_feat += [str(fineline_num)] * scan_count
    else:
        list_feat += [department_des + '_return'] * (- scan_count)
        list_feat += [str(fineline_num) + '_return'] * (- scan_count)
    return ' '.join(list_feat)

df_train['concat_feat'] = df_train.apply(lambda x: concat_feat(x['ScanCount'], x['DepartmentDescription'], x['FinelineNumber']), axis=1)
df_test['concat_feat'] = df_test.apply(lambda x: concat_feat(x['ScanCount'], x['DepartmentDescription'], x['FinelineNumber']), axis=1)

df_grouped = df_train.groupby('VisitNumber', sort=False)
df_train_agg = df_grouped.agg({'TripType': lambda x: x.iloc[0],
                'Weekday': lambda x: x.iloc[0],
                'concat_feat': lambda x: ' '.join(x)}).reset_index()
# df_train_agg['concat_feat'] = df_train_agg.apply(lambda x: x['Weekday'] + ' ' + x['concat_feat'], axis=1)

df_grouped = df_test.groupby('VisitNumber', sort=False)
df_test_agg = df_grouped.agg({'Weekday': lambda x: x.iloc[0],
                              'concat_feat': lambda x: ' '.join(x)}).reset_index()
# df_test_agg['concat_feat'] = df_test_agg.apply(lambda x: x['Weekday'] + ' ' + x['concat_feat'], axis=1)

# transform string to tfidf vector
vectorizer = TfidfVectorizer(use_idf=True)
vectorizer.fit(df_train_agg['concat_feat'].append(df_test_agg['concat_feat']))
x_train = vectorizer.transform(df_train_agg['concat_feat'])
x_test = vectorizer.transform(df_test_agg['concat_feat'])

# transform weekday to tf vector
vectorizer_week = TfidfVectorizer(use_idf=False)
vectorizer_week.fit(df_train_agg['Weekday'].append(df_test_agg['Weekday']))
x_train_week = vectorizer_week.transform(df_train_agg['Weekday'])
x_test_week = vectorizer_week.transform(df_test_agg['Weekday'])

# stack two matrices together
x_train = hstack([x_train, x_train_week], 'csr')
x_test = hstack([x_test, x_test_week], 'csr')

le_triptype = LabelEncoder()
y_train = df_train_agg['TripType']
y_train = le_triptype.fit_transform(y_train)

with open('./data/pack.pkl', 'wb') as f:
    pickle.dump((x_train, y_train, x_test), f)










