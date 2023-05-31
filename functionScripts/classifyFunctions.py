import os
import sys
import pandas as pd
import numpy as np
from copy import deepcopy

from os.path import exists
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import helperFunctions as hf
import plotFunctions as pf

from sklearn import preprocessing, linear_model, cluster, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Used to create a consistent processing pipeline prior to training/testing.
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import FunctionTransformer, RobustScaler, normalize
from sklearn.base import BaseEstimator, TransformerMixin

from mrmr import mrmr_classif

def correlationMatrixPlot(X_data, col_names):

    corrMat = np.corrcoef(X_data, rowvar=False)

    plt.figure(figsize=(40, 40))
    sns.set(font_scale=1)
    sns.heatmap(corrMat, cmap="coolwarm", xticklabels=col_names, yticklabels=col_names)  # , annot=True
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    return corrMat

def extract_sorted_correlations(correlation_matrix, variable_names):
    num_variables = len(variable_names)
    correlation_pairs = []

    for i in range(num_variables):
        for j in range(i+1, num_variables):
            correlation_pairs.append(
                [variable_names[i], variable_names[j], correlation_matrix[i, j]])

    # Sort by the absolute value of the correlation in descending order
    correlation_pairs.sort(key=lambda x: -abs(x[2]))

    return correlation_pairs

def classifySamples(pandasdf, classifyDict, dirDict):

    # ================== Perform Filtration and agglomeration as desired ==================
    filtAggFileName = f"{dirDict['tempDir']}lightsheet_data_filtAgg.pkl"

    if (classifyDict['featurefilt'] or classifyDict['featureAgg']):

        if not exists(filtAggFileName):

            print('Generating filtered and aggregated data file...')
            # Set to filter features based on outliers (contains 99.9th percentile values)
            if classifyDict['featurefilt']:
                pandasdf = hf.filter_features(pandasdf, classifyDict)

            if classifyDict['featureAgg']:
                ls_data_agg = hf.agg_cluster(pandasdf, classifyDict, dirDict)
            else:
                ls_data_agg = pandasdf.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])

            # Save for later
            ls_data_agg.to_pickle(filtAggFileName)

        else:

            # Load from file
            print('Loading filtered and aggregated data from file...')
            ls_data_agg = pd.read_pickle(filtAggFileName)
    else:

        # Reformat data for classification
        ls_data_agg = pandasdf.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])

    # Plot correlations after aggregation
    if 0:
        corr_matrix = np.corrcoef(ls_data_agg.values.T)
        yticklabels = ls_data_agg.columns.tolist()
        plotDim = len(yticklabels) * 12 * 0.015
        plt.figure(figsize=(plotDim*1.1, plotDim))
        sns.heatmap(corr_matrix, cmap='rocket', fmt='.2f', yticklabels=yticklabels, xticklabels=yticklabels, square=True)
        plt.show()
        
    # ================== Shape data for classification ==================

    # X, y, featureNames, numYDict = hf.reformatData(pandasdf, classifyDict)
    X = np.array(ls_data_agg.values)
    # y = np.array([x[0:-1] for x in np.array(ls_data_agg.index)])
    yStr = np.array([x[0:-1] for x in np.array(ls_data_agg.index)])
    yDict = dict(zip(np.unique(yStr), range(1, len(np.unique(yStr))+1)))
    y = np.array([yDict[x] for x in yStr])
    
    featureNames = np.array(ls_data_agg.columns)
    # numYDict = {y: i for i, y in enumerate(np.unique(y))}
    numYDict = {value: key for key, value in yDict.items()}

    # ================== Pipeline Construction ==================
    max_iter = classifyDict['max_iter']
    paramGrid = dict()
    pipelineList = []

    # Create a model with each of the desired feature counts.
    if 'LogReg' in classifyDict['model'] or 'SVM' in classifyDict['model']:
        paramGrid['classif__C'] = classifyDict['pGrid']['classif__C']

    # Select Feature scaling model
    if classifyDict['model_featureScale']:
        scaleMod = RobustScaler()
    else:
        scaleMod = FunctionTransformer(emptyFxn)

    if 'scaleMod' in locals():
        pipelineList.append(('featureScale', scaleMod))

    # Select Feature selection model
    match classifyDict['model_featureSel']:
        case 'MRMR':
            # featureSelMod = FunctionTransformer(MRMRFeatureSelector, kw_args={'n_features_to_select': 10, 'pass_y': True, 'featureSel__feature_names_out': True}, )
            featureSelMod = MRMRFeatureSelector(n_features_to_select=10)
            modVar = 'n_features_to_select'
        case 'Univar':
            featureSelMod = SelectKBest(score_func=f_classif, k='all')
            modVar = 'k'
        case 'mutInfo':
            featureSelMod = SelectKBest(score_func=mutual_info_classif, k='all')
            modVar = 'k'
        case 'RFE':
            featureSelMod = SequentialFeatureSelector(classif, n_features_to_select=10, direction="forward")
            modVar = 'n_features_to_select'
        case 'None':
            featureSelMod = FunctionTransformer(emptyFxn)
        case _:
            ValueError('Invalid model_featureSel')

    if 'featureSelMod' in locals():
        pipelineList.append(('featureSel', featureSelMod))

    # Select Classifying model
    match classifyDict['model']:
        # Logistic Regression Models
        case 'LogRegL2':
            classif = linear_model.LogisticRegression(penalty='l2', multi_class=classifyDict['multiclass'], solver='saga', max_iter=max_iter, dual=False)
        case 'LogRegL1':
            classif = linear_model.LogisticRegression(penalty='l1', multi_class=classifyDict['multiclass'], solver='saga', max_iter=max_iter)
        case 'LogRegElastic':
            classif = linear_model.LogisticRegression(penalty='elasticnet', multi_class=classifyDict['multiclass'], solver='saga', l1_ratio=0.5, max_iter=max_iter)
            paramGrid['classif__l1_ratio'] = classifyDict['pGrid']['classif__l1_ratio']
        case _:
            ValueError('Invalid modelStr')

    pipelineList.append(('classif', classif))

    # Create a base model
    
    clf = Pipeline(pipelineList)
    modelList = []

    # Iterate over the desired feature counts
    if classifyDict['model_featureSel'] != 'None':
        for k in classifyDict['model_featureSel_k']:
            clf_copy = deepcopy(clf)
            modelList.append(clf_copy.set_params(**{f"featureSel__{modVar}": k}))

    else:
        modelList.append(clf)   

    # ================== Classification ==================

    fits = ['Real']
    if classifyDict['shuffle']:
        fits = fits + ['Shuffle']

    for fit in fits:
        if fit == 'Shuffle':
            y = shuffle(y)

        # Generate a confusion matrix to represent classification accuracy. Also creates the PR Curve
        findConfusionMatrix(modelList, X, y, numYDict, featureNames, classifyDict['CV_count'], fit=fit, nestedCVSwitch=classifyDict['gridCV'], paramGrid=paramGrid, dirDict=dirDict)

        # if fit != 'Shuffle' and len(numYDict) != 2:
        #     findConfusionMatrix_LeaveOut(daObj, clf, X, y, numYDict, 8, fit=fit)

def findConfusionMatrix(modelList, X, y, labelDict, featureNames, KfoldNum, fit, nestedCVSwitch=False, paramGrid=None, dirDict=[]):
    # Find the confusion matrix for the model.
    # clf = model, X = Data formated n_samples, n_features, y = n_samples * 1 labels
    # KfoldNum = number of folds to use for cross-validation
    # leaveOut = train the model n_classes number of time, leaving out a class each time, average across the confusion matricies.
    # labelDict = converts from numbered classes to labels.

    # Initialize matricies for the storing features which go into the confusion matrix and the later PR Curve.
    # For each class, store an array of real labels, predicted labels,
    n_classes = len(np.unique(y))

    # For storing the features used to make the classification for each fold.
    YtickLabs = [labelDict[x] for x in np.unique(y)]

    # Initialize arrays for conf_matrix
    skf = StratifiedKFold(n_splits=KfoldNum)  # 8 examples of each label
    skf.get_n_splits(X, y)

    # Based on kFoldNum, pick a number
    # in LOOCV, inner loop is outer loop - 1.
    if KfoldNum == 8:
        innerLoopKFoldNum = 7
    # in 
    elif KfoldNum == 4:
        innerLoopKFoldNum = 3

    # Create a label binarizer
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y_bin = lb.transform(y)
    
    # If the vector is already binary, expand it for formatting consistency.
    if max(y) == 1:
        y_bin = np.hstack([np.abs(y_bin-1), y_bin])

    for clf in modelList:

        # Create a string to represent the model
        modelStr, saveStr = hf.modelStrGen(clf)

        penaltyStr = clf['classif'].penalty
        print(f"evaluating model: {modelStr}")

        # Initialize things
        y_real, y_prob = [], []
        selected_features_list = [[] for _ in range(n_classes)]
        selected_features_params = [[] for _ in range(n_classes)]
        conf_matrix_list_of_arrays = []
        scores = []

        for train_index, test_index in skf.split(X, y):
            # Grab training data and test data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_bin_test = y_bin[test_index]

            if nestedCVSwitch and paramGrid:
                # Do a grid search for the relevant parameters
                grid_search = GridSearchCV(clf, paramGrid, cv=innerLoopKFoldNum, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                clf.set_params(**best_params)

                # Report out on those models
                print({k: v for k, v in grid_search.best_params_.items()})

            # Fit the model
            clf.fit(X_train, y_train)

            # if best K, get ride of the non-label names
            if 'featureSel' in clf.named_steps.keys():
                featureNamesSub = featureNames[clf['featureSel'].get_support(indices=True)]
            else:
                featureNamesSub = featureNames

            # Predict answers for the X_test set
            # if isinstance(clf['classif'], OneVsRestClassifier):
            #     # Reformat the output indicator, since the confusion_matrix fxn doesn't accept it.
            #     x_test_predict = np.argmax(clf.predict(X_test), axis=1)
            # else:
            # create the predictions for the test set based on the model
            x_test_predict = clf.predict(X_test)

            # Extract the confusion matrix for the split
            conf_matrix = confusion_matrix(y_test, x_test_predict)

            # In cases where test set has more than 1 instance of a class, normalize reach row.
            if np.max(np.sum(conf_matrix, axis=1)) != 1:
                sums = np.sum(conf_matrix, axis=1)
                conf_matrix = conf_matrix / sums[:, np.newaxis]

            # Append the confusion matrix to the larger stack
            conf_matrix_list_of_arrays.append(conf_matrix)
            
            # Extract the score, along the diagonal.
            scores.append(np.mean(np.diag(conf_matrix)))

            # Calculate PR Curve by predicting test
            y_scores = clf.predict_proba(X_test)

            y_real.append(y_bin_test)
            y_prob.append(y_scores)

            # Append features List in case of LASSO or ElasticNet Regression
            if penaltyStr != 'l2' and penaltyStr != 'None':

                for idx, coefSet in enumerate(clf._final_estimator.coef_):
                    selected_features = featureNamesSub[coefSet != 0]
                    selected_features_list[idx].append(selected_features)
                    if nestedCVSwitch:
                        selected_features_params[idx].append(best_params)

        # Following the CV, concatonate results across all splits and use to Plot PR Curve
        if fit != 'Shuffle':
            pf.plotPRcurve(n_classes, y_real, y_prob, labelDict, modelStr)

        # Plot Confusion Matrix
        pf.plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, saveStr, dirDict)

        if fit != 'Shuffle' and penaltyStr != 'l2' and penaltyStr != 'None':
            hf.stringReportOut(selected_features_list, selected_features_params, YtickLabs)

def findConfusionMatrix_LeaveOut(daObj, clf, X, y, labelDict, KfoldNum, fit):

    # Repeat the procedure below many times, but you need to have
    # a copy of x and y without the class - go through the procedure
    # Predict based on the full X.
    # Write a special function to reformat the Matrix - expand it to include the left out class.
    # Stack as usual.

    # Perform this procedure the same number of times as there are classes.
    classList = np.unique(np.array(y))

    conf_mat_final = np.zeros((len(classList), len(classList)))

    for idx, class_out in enumerate(classList):
        class_out_ind = y == class_out
        class_in_ind = y != class_out

        # Remove the class examples from the training data
        X_in = X[class_in_ind, :]
        y_in = y[class_in_ind]

        # Initialize arrays for conf_matrix
        skf = StratifiedKFold(n_splits=8)  # 8 examples of each label
        skf.get_n_splits(X_in, y_in)
        conf_matrix_list_of_arrays = []

        for train_index, test_index in skf.split(X_in, y_in):
            # Grab training data and test data
            X_train, X_test = X_in[train_index], X_in[test_index]
            y_train, y_test = y_in[train_index], y_in[test_index]

            # Add the removed data back into the testing set.
            X_test_new = np.concatenate((X_test, X[class_out_ind, :]), axis=0)
            y_test_new = np.concatenate((y_test, y[class_out_ind]), axis=0)

            # Fit the model
            clf.fit(X_train, y_train)

            # Extract the confusion matrix
            conf_matrix = confusion_matrix(y_test_new, clf.predict(X_test_new))

            # Normalize the conf_matrix, since the left out class has many more testing examples
            # conf_matrix = normalize(conf_matrix, axis=1)

            conf_matrix_list_of_arrays.append(conf_matrix)

        conf_matrix_list_of_arrays = np.array(conf_matrix_list_of_arrays)

        # once finished with the class, create the row in the confusion matrix it represents by averaging and extracting.
        # mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
        sum_of_conf_matrix_arrays = np.sum(conf_matrix_list_of_arrays, axis=0)

        # Put into final confusion matrix
        conf_mat_final[idx, :] = sum_of_conf_matrix_arrays[idx, :]

    # Normalize each row
    row_sums = conf_mat_final.sum(axis=1, keepdims=True)
    conf_mat_final = conf_mat_final / row_sums

    # Plotting and reporting.
    YtickLabs = [labelDict[x] for x in classList]
    plt.figure(figsize=(8, 8))

    ax = sns.heatmap(conf_mat_final, linewidth=0.25, cmap='coolwarm', annot=True, fmt=".2f")
    ax.set(xticklabels=YtickLabs, yticklabels=YtickLabs, xlabel='Predicted Label', ylabel='True Label')

    # cbar_kws=dict(set_over=1.0)
    plt.title(fit + ', LOO: ' + str(daObj))
    plt.show()

def emptyFxn(X):
    return X

class MRMRFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=10):
        self.n_features_to_select = n_features_to_select
    
    def fit(self, X, y=None):

        # Transform X and Y into dataframes
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=['y'])

        yDict = {x: i for i, x in enumerate(np.unique(y))}
        y_df_int = y_df.replace(yDict)

        selected_features = mrmr_classif(X=X_df, y=y_df_int, K=self.n_features_to_select, show_progress=False)
        selected_features = np.array(selected_features)
        # print(f"features selected: {selected_features}")

        X_df = X_df.values
        y_df_int = y_df_int.values.ravel()

        # store features selected in self for later transformation
        self.selected_features_ = selected_features

        return self
    
    def get_support(self, indices=True):
        return self.selected_features_
    
    def transform(self, X, y=None):
        # Pick out previously selected variables
        # feature_to_use = self.selected_features_[0:self.n_features_to_select]
        X_fit = X[:, self.selected_features_]
        # print(f"transformed X: {X_fit.shape}")

        return X_fit