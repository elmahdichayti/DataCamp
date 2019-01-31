import os
import pandas as pd
import numpy as np
import rampwf as rw
from get_data import get_data


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#-----------------------------------------------------------------------
problem_title = 'Electric grid imbalances'
_target_column_name = 'imbalance'
_prediction_label_names = []

# A type which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

score_types = [
    rw.score_types.normalized_rmse.NormalizedRMSE(name="normalized_rmse", precision=3),
    rw.score_types.mare.MARE(name="mare", precision=3),
]

#-----------------------------------------------------------------------
def get_cv(X, y):
    """Returns stratified randomized folds."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    return cv.split(X,y)

def _read_data(path, f_name):
    data = pd.read_excel(os.path.join(path, 'data', f_name), index_col="time").drop("time.1", axis=1)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        return X_df[:10000], y_array[:10000]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'df_train.xlsx'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'df_test.xlsx'
    return _read_data(path, f_name)
