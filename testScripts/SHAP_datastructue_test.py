import pickle
import os
import sys
import numpy as np

fileNames = ['shap_values', 'X_train', 'featureNames']

with open(f'testScripts/SHAP_test_file_shap_values.pkl', 'rb') as file:
    shap_values = pickle.load(file)

with open(f'testScripts/SHAP_test_file_X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open(f'testScripts/SHAP_test_file_featureNames.pkl', 'rb') as file:
    featureNames = pickle.load(file)

# Prepare the test case - remove some random feature from copies of each variable
shap_values_A = shap_values.copy()
X_train_A = X_train.copy()
featureNames_A = featureNames.copy()

shap_values_B = shap_values.copy()
X_train_B = X_train.copy()
featureNames_B = featureNames.copy()

a_var_remove = 10
b_var_remove = 20

shap_values_A = np.hstack([shap_values_A[:, 0:a_var_remove], shap_values_A[:, a_var_remove+1:]])
X_train_A = np.hstack([X_train_A[:, 0:a_var_remove], X_train_A[:, a_var_remove+1:]])
featureNames_A = np.concatenate([featureNames_A[0:a_var_remove], featureNames_A[a_var_remove+1:]])
print(f'Shape of shap_values_A: {shap_values_A.shape}')
print(f'Shape of X_train_A: {X_train_A.shape}')
print(f'Shape of featureNames_A: {featureNames_A.shape}')

shap_values_B = np.hstack([shap_values_B[:, 0:b_var_remove], shap_values_B[:, b_var_remove+1:]])
X_train_B = np.hstack([X_train_B[:, 0:b_var_remove], X_train_B[:, b_var_remove+1:]])
featureNames_B = np.concatenate([featureNames_B[0:b_var_remove], featureNames_B[b_var_remove+1:]])
print(f'Shape of shap_values_B: {shap_values_B.shape}')
print(f'Shape of X_train_B: {X_train_B.shape}')
print(f'Shape of featureNames_B: {featureNames_B.shape}')

# Goal - combine _A and _B structures in such a way that SHAP functions work, entries aren't duplicated.

# Start by collecting a single complete feature list
featureNames_C = set(np.concatenate([featureNames_A, featureNames_B]))

# Some feature have 2x the data of other features...

