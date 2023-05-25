CV_repeats = 10
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import pandas as pd
import sys
sys.path.append('../LSP_Repo/')

import classifyFunctions
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, linear_model

random_states = np.arange(10)

for i, CV_repeat in enumerate(range(CV_repeats)): 
    #Verbose 
    print('\n------------ CV Repeat number:', CV_repeat)
    #Establish CV scheme
    CV = StratifiedKFold(n_splits=8, shuffle=True, random_state=random_states[i]) # Set random state 

    ix_training, ix_test = [], []
    # Loop through each fold and append the training & test indices to the empty lists above
    for fold in CV.split(df):
        ix_training.append(fold[0]), ix_test.append(fold[1])
        
    ## Loop through each outer fold and extract SHAP values 
    for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)): 
        #Verbose
        print('\n------ Fold Number:',i)
        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]
        
        ## Establish inner CV for parameter optimization #-#-#
        cv_inner = StratifiedKFold(n_splits=7, shuffle=True, random_state=1) #-#-#
        
        pipeline = make_pipeline(preprocessing.RobustScaler(), model)
        pipelineT = make_pipeline(preprocessing.RobustScaler())
        pipelineT.fit(X_train, y_train)
        pipeline.fit(X_train, y_train)

        # Search to optimize hyperparameters
        model = RandomForestRegressor(random_state=10)
        search = RandomizedSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=cv_inner, refit=True) #-#-#
        result = search.fit(X_train, y_train) #-#=#

        # Fit model on training data 
        result.best_estimator_.fit(X_train, y_train) #-#-#
    
        # Use SHAP to explain predictions using best estimator 
        explainer = shap.TreeExplainer(result.best_estimator_) 
        shap_values = explainer.shap_values(X_test)

        # Extract SHAP information per fold per sample 
        for i, test_index in enumerate(test_outer_ix):
            shap_values_per_cv[test_index][CV_repeat] = shap_values[i] 