import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from sklearn.datasets import make_classification
from mrmr import mrmr_classif

# X, y = make_classification(n_samples = 64, n_features = 100, n_informative = 10, n_redundant = 40, n_classes=8)
# X = pd.DataFrame(X)
# y = pd.Series(y)

def test_mrmr(df):

    df_sc = df.sample(frac=1)
    y_df = np.array([x[0:-1] for x in np.array(df_sc.index)])
    featureNames = df_sc.columns
    yDict = {x: i for i, x in enumerate(np.unique(y_df))}

    X_df = pd.DataFrame(df_sc.reset_index(drop=True), columns=featureNames).sample(frac=1).reset_index(drop=True)
    X_df = RobustScaler().fit_transform(X_df)
    X_df = pd.DataFrame(X_df, columns=featureNames)

    y_df = pd.DataFrame(y_df, columns=['y'])
    y_df_int = y_df.replace(yDict)

    selected_features = mrmr_classif(X=X_df, y=y_df_int, K=20)
    print(selected_features)