import pandas as pd
import numpy as np


def compare(result1, result2):
    df_1 = pd.read_csv('../data/models/{}.csv'.format(result1))
    df_2 = pd.read_csv('../data/models/{}.csv'.format(result2))

    corr_coef = 0

    for i in range(1, 38):
        corr_coef += np.corrcoef(df_1.iloc[:, i], df_2.iloc[:, i], rowvar=0)[0, 1]

    print 'corr_coef of {} to {} is: {}'.format(result1, result2, corr_coef)
    return corr_coef