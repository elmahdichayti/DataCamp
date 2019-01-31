
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
from scipy.special import boxcox1p

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        
        season_dico = {1:"Winter", 2:"Spring", 3:"Summer", 4:"Fall"}
        X_df_new["Season"] = X_df_new.Month.apply(lambda x: season_dico[(x%12 + 3)//3])

        X_df_new["Weekend"] = X_df_new.index.weekday
        X_df_new["Weekend"] = X_df_new["Weekend"].apply(lambda t: 1 if t >= 5 else 0)
        
        X_df_new["Year"] = X_df_new.index.year
        
        X_df_new = pd.get_dummies(X_df_new, columns=["Season", "Month"], drop_first=True)
        
        X_df_new = compute_rolling_std(X_df_new, 'Total wind generation (W)', '24h')

        X_df_new = compute_rolling_mean(X_df_new, 'Total photovoltaic production (W)', '24h')
        X_df_new = compute_rolling_mean(X_df_new, 'Total wind generation (W)', '24h')
        
        X_df_new = X_df_new.drop(["Pseudo radiation"], axis=1)
        
        return X_df_new
    

def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    return data


def compute_rolling_mean(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the mean over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'mean'])
    data[name] = data[feature].rolling(time_window, center=center).mean()
    data[name] = data[name].ffill().bfill()
    return data