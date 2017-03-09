# Author: Jiang Hao <nju.jianghao@foxmail.com>

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy import sparse

# read data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_train.fillna({'Upc': 0, 'DepartmentDescription': 'UNKNOWN', 'FinelineNumber': 10000}, inplace=True)
df_test.fillna({'Upc': 0, 'DepartmentDescription': 'UNKNOWN', 'FinelineNumber': 10000}, inplace=True)

# list controversial punctuations
DEPARTMENT_NAMES = {
                '1-HR PHOTO': '1HRPHOTO',
                'BRAS & SHAPEWEAR': 'BRASANDSHAPEWEAR',
                'CANDY, TOBACCO, COOKIES': 'CANDYTOBACCOCOOKIES',
                'GIRLS WEAR, 4-6X  AND 7-14': 'GIRLSWEARFOURSIX',
                'HOUSEHOLD CHEMICALS/SUPP': 'HOUSEHOLDCHEMICALSSUPP',
                'MEAT - FRESH & FROZEN': 'MEATFRESHANDFROZEN',
                'LIQUOR,WINE,BEER': 'LIQUORWINEBEER',
                'OPTICAL - FRAMES': 'OPTICALFRAMES',
                'OPTICAL - LENSES': 'OPTICALLENSES',
                'SLEEPWEAR/FOUNDATIONS': 'SLEEPWEARFOUNDATIONS',
                'SWIMWEAR/OUTERWEAR': 'SWIMWEAROUTERWEAR'}
# filter the stop punctuations
def department_map(raw_text):
    if raw_text in DEPARTMENT_NAMES.keys():
        return DEPARTMENT_NAMES[raw_text]
    else:
        return raw_text
df_train['DepartmentDescription'] = df_train['DepartmentDescription'].apply(lambda x: department_map(x))
df_test['DepartmentDescription'] = df_test['DepartmentDescription'].apply(lambda x: department_map(x))

def concat_feat(scan_count, department_des, fineline_num):
    department_des = ''.join(department_des.split())
    list_department_buy_feat = []
    list_fineline_buy_feat = []
    list_department_ret_feat = []
    list_fineline_ret_feat = []
    num_buy = 0
    num_return = 0
    if scan_count > 0:
        list_department_buy_feat = [department_des] * scan_count
        list_fineline_buy_feat = [str(int(fineline_num))] * scan_count
        num_buy += scan_count
    else:
        list_department_ret_feat = [department_des + 'return'] * (- scan_count)
        list_fineline_ret_feat = [str(int(fineline_num)) + 'return'] * (- scan_count)
        num_return += (-scan_count)
    dict_return = {'depart_buy': ' '.join(list_department_buy_feat), 'fineline_buy': ' '.join(list_fineline_buy_feat), 'depart_return': ' '.join(list_department_ret_feat), 'fineline_return': ' '.join(list_fineline_ret_feat), 'num_buy': num_buy, 'num_return': num_return}
    return dict_return

def count_kind(raw_text):
    return len(set(raw_text.split()))

# add 6 columns: 'depart_buy', 'fineline_buy', 'depart_return', 'fineline_return', 'num_buy', 'num_return'
df_temp = pd.DataFrame(list(df_train.apply(lambda x: concat_feat(x['ScanCount'], x['DepartmentDescription'], x['FinelineNumber']), axis=1)))
df_train = pd.concat([df_train, df_temp], axis=1)
df_temp = pd.DataFrame(list(df_test.apply(lambda x: concat_feat(x['ScanCount'], x['DepartmentDescription'], x['FinelineNumber']), axis=1)))
df_test = pd.concat([df_test, df_temp], axis=1)

df_grouped = df_train.groupby('VisitNumber', sort=False)
df_train_agg = df_grouped.agg({
                'TripType': lambda x: x.iloc[0],
                'Weekday': lambda x: x.iloc[0],
                'depart_buy': lambda x: ' '.join(x),
                'fineline_buy': lambda x: ' '.join(x),
                'depart_return': lambda x: ' '.join(x),
                'fineline_return': lambda x: ' '.join(x),
                'num_buy': np.sum,
                'num_return': np.sum}).reset_index()
# df_train_agg['concat_feat'] = df_train_agg.apply(lambda x: x['Weekday'] + ' ' + x['concat_feat'], axis=1)

df_grouped = df_test.groupby('VisitNumber', sort=False)
df_test_agg = df_grouped.agg({
                'Weekday': lambda x: x.iloc[0],
                'depart_buy': lambda x: ' '.join(x),
                'fineline_buy': lambda x: ' '.join(x),
                'depart_return': lambda x: ' '.join(x),
                'fineline_return': lambda x: ' '.join(x),
                'num_buy': np.sum,
                'num_return': np.sum}).reset_index()
# df_test_agg['concat_feat'] = df_test_agg.apply(lambda x: x['Weekday'] + ' ' + x['concat_feat'], axis=1)

# label encode Weekday
le_weekday = LabelEncoder()
week_train = le_weekday.fit(df_train_agg['Weekday'].append(df_test_agg['Weekday']))
week_train = le_weekday.transform(df_train_agg['Weekday']).reshape((-1, 1))
week_test = le_weekday.transform(df_test_agg['Weekday']).reshape((-1, 1))

# num of buy kinds and num of return kinds
kindbuy_train = df_train_agg['depart_buy'].apply(lambda x: count_kind(x))
kindret_train = df_train_agg['depart_return'].apply(lambda x: count_kind(x))
onlybuy_train = (kindret_train == 0).apply(lambda x: 1 if x is True else 0)
onlyret_train = (kindbuy_train == 0).apply(lambda x: 1 if x is True else 0)
kindbuy_train = np.array(kindbuy_train).reshape((-1, 1))
kindret_train = np.array(kindret_train).reshape((-1, 1))
onlybuy_train = np.array(onlybuy_train).reshape((-1, 1))
onlyret_train = np.array(onlyret_train).reshape((-1, 1))
kindbuy_test = df_test_agg['depart_buy'].apply(lambda x: count_kind(x))
kindret_test = df_test_agg['depart_return'].apply(lambda x: count_kind(x))
onlybuy_test = (kindret_test == 0).apply(lambda x: 1 if x is True else 0)
onlyret_test = (kindbuy_test == 0).apply(lambda x: 1 if x is True else 0)
kindbuy_test = np.array(kindbuy_test).reshape((-1, 1))
kindret_test = np.array(kindret_test).reshape((-1, 1))
onlybuy_test = np.array(onlybuy_test).reshape((-1, 1))
onlyret_test = np.array(onlyret_test).reshape((-1, 1))

# num buy and num return
numbuy_train = np.array(df_train_agg['num_buy']).reshape((-1, 1))
numbuy_test = np.array(df_test_agg['num_buy']).reshape((-1, 1))
numret_train = np.array(df_train_agg['num_return']).reshape((-1, 1))
numret_test = np.array(df_test_agg['num_return']).reshape((-1, 1))

# transform department_buy feat to percentage using L1 normalization
vectorizer_department_buy = TfidfVectorizer(use_idf=False, norm='l1')
vectorizer_department_buy.fit(df_train_agg['depart_buy'].append(df_test_agg['depart_buy']))
m_train_department_buy = vectorizer_department_buy.transform(df_train_agg['depart_buy'])
m_test_department_buy = vectorizer_department_buy.transform(df_test_agg['depart_buy'])

# transform department_return feat to percentage using L1 normalization
vectorizer_department_return = TfidfVectorizer(use_idf=False, norm='l1')
vectorizer_department_return.fit(df_train_agg['depart_return'].append(df_test_agg['depart_return']))
m_train_department_return = vectorizer_department_return.transform(df_train_agg['depart_return'])
m_test_department_return = vectorizer_department_return.transform(df_test_agg['depart_return'])

# transform fineline_buy feat to percentage using L1 normalization
vectorizer_fineline_buy = TfidfVectorizer(use_idf=False, norm='l1')
vectorizer_fineline_buy.fit(df_train_agg['fineline_buy'].append(df_test_agg['fineline_buy']))
m_train_fineline_buy = vectorizer_fineline_buy.transform(df_train_agg['fineline_buy'])
m_test_fineline_buy = vectorizer_fineline_buy.transform(df_test_agg['fineline_buy'])

# transform fineline_return feat to percentage using L1 normalization
vectorizer_fineline_return = TfidfVectorizer(use_idf=False, norm='l1')
vectorizer_fineline_return.fit(df_train_agg['fineline_return'].append(df_test_agg['fineline_return']))
m_train_fineline_return = vectorizer_fineline_return.transform(df_train_agg['fineline_return'])
m_test_fineline_return = vectorizer_fineline_return.transform(df_test_agg['fineline_return'])

# hstack all the features
'''
feat_train: 7 cols
m_train_department_buy 83 cols
m_train_department_return 81 cols
m_train_fineline_buy 5323 cols
m_train_fineline_return 3936 cols
'''
feat_train = np.hstack((week_train, numbuy_train, numret_train, kindbuy_train, kindret_train, onlybuy_train, onlyret_train))
feat_test = np.hstack((week_test, numbuy_test, numret_test, kindbuy_test, kindret_test, onlybuy_test, onlyret_test))
feat_train = sparse.csr_matrix(feat_train)
feat_test = sparse.csr_matrix(feat_test)

# only thi part need change
x_train = hstack((m_train_department_buy, m_train_department_return, m_train_fineline_buy, m_train_fineline_return), 'csr')
x_test = hstack((m_test_department_buy, m_test_department_return, m_test_fineline_buy, m_test_fineline_return), 'csr')

# label encoder trip type
le_triptype = LabelEncoder()
y_train = df_train_agg['TripType']
y_train = le_triptype.fit_transform(y_train)

# dump to file
with open('./data/pack.pkl', 'wb') as f:
    pickle.dump((x_train, y_train, x_test), f)
