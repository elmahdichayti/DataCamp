# -*- coding: utf-8 -*-
import pandas as pd

def get_data(train=True):
    """
    This function fetches the data from the github, in the form of a tuple (X,y)
    :param train :boolean, if True, will return the training data, else, will return the test data.
    return       : tuple (X,y)
    """
    if train :
        path = 'data/df_train.xlsx'
    else :
        path = 'data/df_test.xlsx'

    df = pd.read_excel(path, index_col="time")
    df = df.drop("time.1", axis=1)

    return (df.drop('imbalance',axis=1), df['imbalance'])
