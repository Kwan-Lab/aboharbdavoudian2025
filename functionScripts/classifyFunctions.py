import os
import sys
import pandas as pd
import numpy as np
from os.path import exists
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import helperFunctions as hf
import plotFunctions as pf


def correlationMatrixPlot(X_data, col_names):
    import numpy as np
    import seaborn as sns

    corrMat = np.corrcoef(X_data, rowvar=False)

    plt.figure(figsize=(40, 40))
    sns.set(font_scale=1)
    sns.heatmap(corrMat, cmap="coolwarm", xticklabels=col_names,
                yticklabels=col_names)  # , annot=True
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


def iterativeCorrShrinkage(X_data, col_names, threshold=0.9):
    import numpy as np
    import seaborn as sns

    corrMat = np.corrcoef(X_data, rowvar=False)

    num_variables = len(col_names)
    correlation_pairs = []

    for i in range(num_variables):
        for j in range(i+1, num_variables):
            correlation_pairs.append(
                [col_names[i], col_names[j], corrMat[i, j]])

    # Sort by the absolute value of the correlation in descending order
    correlation_pairs.sort(key=lambda x: -abs(x[2]))

    # Go through, finding
    features_to_remove = []
    high_corr_cleared = True

    while high_corr_cleared:
        max_corr_pair = correlation_pairs[0]

        if abs(max_corr_pair[2]) >= threshold:
            features_to_remove.append(max_corr_pair[1])
            correlation_pairs = [pair for pair in correlation_pairs if pair[0]
                                 != max_corr_pair[1] and pair[1] != max_corr_pair[1]]
        else:
            high_corr_cleared = False

    # Remove the features from the data and return the new list
    remaining_col_idx = [i for i, name in enumerate(
        col_names) if name not in features_to_remove]

    return remaining_col_idx


def classifySamples(pandasdf, classifyDict, dirDict):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    # Used to create a consistent processing pipeline prior to training/testing.
    from sklearn import preprocessing, linear_model, cluster, svm
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay
    from sklearn.utils import shuffle
    from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector
    from sklearn.preprocessing import FunctionTransformer
    from copy import deepcopy

    # ================== Perform Filtration and agglomeration as desired ==================
    filtAggFileName = f"{dirDict['tempDir']}lightsheet_data_filtAgg.pkl"

    if not exists(filtAggFileName):

        print('Generating filtered and aggregated data file...')
        # Set to filter features based on outliers (contains 99.9th percentile values)
        if classifyDict['featureSel_filter']:
            pandasdf = hf.filter_features(pandasdf, classifyDict)

        if classifyDict['featureSel_agg']:
            ls_data_agg = hf.agg_cluster(pandasdf, classifyDict, dirDict)
        else:
            ls_data_agg = pandasdf.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])

        # Save for later
        ls_data_agg.to_pickle(filtAggFileName)

    else:

        # Load from file
        print('Loading filtered and aggregated data from file...')
        ls_data_agg = pd.read_pickle(filtAggFileName)


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

    # correlation_pairs = extract_sorted_correlations(corrMat, featureNames)

    # ================== Pipeline Construction ==================
    max_iter = 200
    paramGrid = dict()

    # Create a model with each of the desired feature counts.
    pipelineModules = {}

    if 'LogReg' in classifyDict['model'] or 'SVM' in classifyDict['model']:
        paramGrid['cVals'] = [0.001, 0.01, 0.1, 1, 3, 5, 10]

    # Select Classifying model
    match classifyDict['model']:
        # Logistic Regression Models
        case 'LogRegL2':
            classif = linear_model.LogisticRegression(penalty='l2', multi_class='multinomial', solver='saga', max_iter=max_iter, dual=False)
        case 'LogRegL1':
            classif = linear_model.LogisticRegression(penalty='l1', multi_class='multinomial', solver='saga', max_iter=max_iter)
        case 'LogRegElastic':
            classif = linear_model.LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=max_iter)
            paramGrid['l1Vals'] = [0, 0.1, 0.5, 0.9, 1]
        case _:
            ValueError('Invalid modelStr')


    # Select Feature selection model
    match classifyDict['model_featureSel']:
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

    # Create a base model
    clf = Pipeline([('featureSel', featureSelMod), ('classif', classif)])
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
        findConfusionMatrix(modelList, X, y, numYDict, featureNames, 8, fit=fit, nestedCVSwitch=classifyDict['gridCV'], paramGrid=paramGrid, dirDict=dirDict)

        # if fit != 'Shuffle' and len(numYDict) != 2:
        #     findConfusionMatrix_LeaveOut(daObj, clf, X, y, numYDict, 8, fit=fit)


def nestedCV_gridSearch(clf, X_train, y_train, paramGrid, gridSearchCVNum):
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier

    # Set the grid search values if they're going to be used.
    cString = next((key for key in clf.get_params().keys() if 'C' in key), None)
    l1String = next((key for key in clf.get_params().keys() if 'l1_ratio' in key), None)

    # Create a dictionary of the parameters to search over
    penalty_key = next((key for key in clf.get_params().keys() if 'penalty' in key), None)
    if penalty_key is not None:
        penaltyStr = clf.get_params()[penalty_key]
    else:
        penaltyStr = None

    paramGridPass = dict()
    paramGridPass[cString] = paramGrid['cVals']
    if penaltyStr == 'elasticnet':
        paramGridPass[l1String] = paramGrid['l1Vals']

    # Perform Grid Search
    # Do a grid search for the relevant parameters
    grid_search = GridSearchCV(clf, paramGridPass, cv=gridSearchCVNum, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Report out on those models
    print({k: v for k, v in grid_search.best_params_.items()})

    # Set the best parameters
    clf.set_params(**grid_search.best_params_)

    return clf, grid_search.best_params_


def findConfusionMatrix(modelList, X, y, labelDict, featureNames, KfoldNum, fit, nestedCVSwitch=False, paramGrid=None, dirDict=[]):
    # Find the confusion matrix for the model.
    # clf = model, X = Data formated n_samples, n_features, y = n_samples * 1 labels
    # KfoldNum = number of folds to use for cross-validation
    # leaveOut = train the model n_classes number of time, leaving out a class each time, average across the confusion matricies.
    # labelDict = converts from numbered classes to labels.

    import numpy as np
    from sklearn import preprocessing
    from sklearn.metrics import confusion_matrix
    from sklearn.multiclass import OneVsRestClassifier
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SequentialFeatureSelector

    # Initialize matricies for the storing features which go into the confusion matrix and the later PR Curve.
    # For each class, store an array of real labels, predicted labels,
    n_classes = len(np.unique(y))

    # For storing the features used to make the classification for each fold.
    YtickLabs = [labelDict[x] for x in np.unique(y)]

    # Initialize arrays for conf_matrix
    skf = StratifiedKFold(n_splits=KfoldNum)  # 8 examples of each label
    skf.get_n_splits(X, y)

    # Based on kFoldNum, pick a number
    if KfoldNum == 8:
        innerLoopKFoldNum = 7
    elif KfoldNum == 4:
        innerLoopKFoldNum = 5

    # Create a label binarizer
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y_bin = lb.transform(y)
    
    # If the vector is already binary, expand it for formatting consistency.
    if max(y) == 1:
        y_bin = np.hstack([np.abs(y_bin-1), y_bin])

    for clf in modelList:

        # Create a string to represent the model
        if isinstance(clf['featureSel'], SelectKBest):
            modelStr = f"{str(clf['classif'])}, BestK, k = {clf['featureSel'].k}"
        elif isinstance(clf['featureSel'], SequentialFeatureSelector):
            modelStr = f"{str(clf['classif'])}, SFS, k = {clf['featureSel'].n_features_to_select}"

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
                # Perform nested CV to find the best parameters
                clf, bestParams = nestedCV_gridSearch(clf, X_train, y_train, paramGrid, innerLoopKFoldNum)

            # Fit the model
            clf.fit(X_train, y_train)

            # if best K, get ride of the non-label names
            if 'featureSel' in clf.named_steps.keys():
                featureNamesSub = featureNames[clf['featureSel'].get_support(indices=True)]
            else:
                featureNamesSub = featureNames

            # PRedict answers for the X_test set
            if isinstance(clf['classif'], OneVsRestClassifier):
                # Reformat the output indicator, since the confusion_matrix fxn doesn't accept it.
                x_test_predict = np.argmax(clf.predict(X_test), axis=1)
            else:
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

                if isinstance(clf['classif'], OneVsRestClassifier):
                    # Loop over each model in the 'OneVsRestClassifier'
                    for idx, coef in enumerate(clf.named_steps['onevsrestclassifier'].estimators_):
                        selected_features = featureNamesSub[coef.coef_[0] != 0]
                        selected_features_list[idx].append(selected_features)
                        if nestedCVSwitch:
                            selected_features_params[idx].append(bestParams)
                else:
                    # Loop over each model in the 'OneVsRestClassifier'
                    for idx, coefSet in enumerate(clf._final_estimator.coef_):
                        selected_features = featureNamesSub[coefSet != 0]
                        selected_features_list[idx].append(selected_features)
                        if nestedCVSwitch:
                            selected_features_params[idx].append(bestParams)

        # Following the CV, concatonate results across all splits and use to Plot PR Curve
        if fit != 'Shuffle':
            pf.plotPRcurve(n_classes, y_real, y_prob, labelDict, modelStr)

        # Plot Confusion Matrix
        pf.plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, modelStr, dirDict)

        # Report on which features make the cut.
        if fit != 'Shuffle' and penaltyStr != 'l2' and penaltyStr != 'None':
            for idx, drug in enumerate(YtickLabs):
                regionList = np.concatenate(selected_features_list[idx])

                # Process the feature per model list into a string
                featurePerModelStr = str([len(x) for x in selected_features_list[0]])
                paramStr = ''

                keyList = selected_features_params[idx][0].keys()
                for key in list(keyList):
                    keyVals = [x[key] for x in selected_features_params[idx]]
                    paramStr += f"{key}: {str(keyVals)} \n"

                if len(regionList) == 0:
                    continue

                regionDict = dict(Counter(regionList))
                labels, counts = list(regionDict.keys()), list(regionDict.values())

                finalStr = hf.conciseStringReport(labels, counts)

                print(f'==== {drug} ==== \n Features per Model: {featurePerModelStr}')
                print(f'Parameters: \n {paramStr}')
                print(f'Total Regions = {str(len(labels))} \n {finalStr}')

def findConfusionMatrix_PerDrug(modelList, X, y, labelDict, featureNames, KfoldNum, fit, nestedCVSwitch=False, paramGrid=None, dirDict=[]):
    # Find the confusion matrix for the model.
    # clf = model, X = Data formated n_samples, n_features, y = n_samples * 1 labels
    # KfoldNum = number of folds to use for cross-validation
    # leaveOut = train the model n_classes number of time, leaving out a class each time, average across the confusion matricies.
    # labelDict = converts from numbered classes to labels.

    import numpy as np
    from sklearn import preprocessing
    from sklearn.metrics import confusion_matrix
    from sklearn.multiclass import OneVsRestClassifier
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold
    
    # Initialize matricies for the storing features which go into the confusion matrix and the later PR Curve.
    # For each class, store an array of real labels, predicted labels,
    n_classes = len(np.unique(y))
    y_real, y_prob = [], []

    # For storing the features used to make the classification for each fold.
    YtickLabs = [labelDict[x] for x in np.unique(y)]

    # Initialize arrays for conf_matrix
    conf_matrix_list_of_arrays = []
    scores = []
    skf = StratifiedKFold(n_splits=KfoldNum)  # 8 examples of each label
    
    

    for clf in modelList:
        
        # Estimator name
        modelStr = f"{str(clf['classif'])}, k = {clf['featureSel'].n_features_to_select}"
        penaltyStr = clf['classif'].penalty

        print(f"evaluating model: {modelStr}")

        # Pull the relevant labels for this drug
        skf.get_n_splits(X, y)
        y = drug_vec

        for train_index, test_index in skf.split(X, y):
            # Grab training data and test data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Perform nested CV to find the best parameters
            if nestedCVSwitch and paramGrid:
                print('Performing nested CV to find the best parameters')
                clf = nestedCV_gridSearch(clf, X_train, y_train, paramGrid, innerLoopKFoldNum)

            # Fit the model
            clf.fit(X_train, y_train)

            # if best K, get ride of the non-label names
            if 'featureSel' in clf.named_steps.keys():
                featureNamesSub = featureNames[clf['featureSel'].get_support(indices=True)]
            else:
                featureNamesSub = featureNames

            # PRedict answers for the X_test set
            if isinstance(clf['classif'], OneVsRestClassifier):
                # Reformat the output indicator, since the confusion_matrix fxn doesn't accept it.
                x_test_predict = np.argmax(clf.predict(X_test), axis=1)
            else:
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

        # Following the CV, concatonate results across all splits and use to Plot PR Curve
        if fit != 'Shuffle':
            pf.plotPRcurve(n_classes, y_real, y_prob, labelDict, modelStr)

        # Plot Confusion Matrix
        pf.plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, modelStr, dirDict)


def findConfusionMatrix_LeaveOut(daObj, clf, X, y, labelDict, KfoldNum, fit):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    # Used to create a consistent processing pipeline prior to training/testing.
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import normalize

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

def CV_feature_selection(pandasdf, classifyDict, dirDict):
    """
    A function which iteratives removes features from a model, and performs cross validated scoring of the model
    Args:
        pandasdf: Pandas dataframe of the data to be classified.
        classifyDict: Dictionary containing information for classification.
            'model_featureSel': String specifying the model to use for feature selection.
            'model_classify': String specifying the model to use for classification.
            'shuffle': Boolean specifying if samples should be shuffled.
            'remove_high_corr': Boolean specifying if high correlation features should be removed.
            'corrThreshold': Float specifying the correlation threshold to remove features at.
        dirDict: a dictionary containing directories for storing any output files
            'classifyDir': the root file for all all downstream files or nested directories
    """
    from sklearn import preprocessing, linear_model
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, mutual_info_classif
    from sklearn.metrics import confusion_matrix
    from plotFunctions import create_heatmaps_allC, create_heatmaps_perDrug
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

    # Add in new folder
    folderTag = 'CV_feature_selection//'
    if folderTag not in dirDict['classifyDir']:
        dirDict['classifyDir'] = dirDict['classifyDir'] + folderTag

    if not os.path.exists(dirDict['classifyDir']):
        os.mkdir(dirDict['classifyDir'])

    # Temp hardcoding of some variables
    max_iter = 500
    # Used to scan over the sparse coefficient space for L1 based feature selection.
    C_vec = [100, 10, 5, 1, 0.5, 0.1]  # , 0.05, 0.01
    multi_class_str = 'ovr'    # Can be ovr or multinomial

    # Reformat the data
    X, y, featureNames, numYDict = hf.reformatData(pandasdf, 'Region_Name', classifyDict) # 'Region_Name', 'abbreviation'
    
    drugNames = list(numYDict.values())

    # If you want to remove high correlation features, do so here
    # if classifyDict['remove_high_corr']:
    #     keep_ind = iterativeCorrShrinkage(X, featureNames, threshold=classifyDict['corrThreshold'])
    #     X = X[:, keep_ind]
    #     featureNames = featureNames[keep_ind]
        
    # Create the feature selection component of the model
    # Maybe Random forest later

    if classifyDict['model_featureSel'] == 'L1':
        featureSelMod = linear_model.LogisticRegression(multi_class=multi_class_str, penalty='l1', solver='liblinear', max_iter=max_iter)
    elif classifyDict['model_featureSel'] == 'Univar':
        featureSelMod = SelectKBest(score_func=f_classif, k='all')
    elif classifyDict['model_featureSel'] == 'mutInfo':
        featureSelMod = SelectKBest(score_func=mutual_info_classif, k='all')
    else:
        KeyError()

    sfm = SelectFromModel(featureSelMod, threshold=1e-6, prefit=False)

    # Create the classification model
    if classifyDict['model_classify'] == 'L2':
        classMod = linear_model.LogisticRegression(penalty='l2', solver='liblinear', max_iter=max_iter, dual=True)

    # Fold these together into a pipeline with proper preprocessing
    pipelineObj = make_pipeline(preprocessing.RobustScaler(), sfm, classMod)

    # Initialize arrays for storing outputs
    n_classes = len(np.unique(y))
    C_feature_sel = [[] for _ in range(len(C_vec))]
    C_feature_coef = [[] for _ in range(len(C_vec))]
    C_conf_mat = [[] for _ in range(len(C_vec))]
    C_scores = [[] for _ in range(len(C_vec))]

    # Identify the estimator's C
    C_param = hf.check_phrase_in_keys('selectfrommodel*C', pipelineObj.get_params())
    C_coef_mat = []

    skf = StratifiedKFold(n_splits=8)  # 8 examples of each label

    # Set the outer loop for selecting C
    for C_idx, C_val in enumerate(C_vec):

        # Set up storage for variables related to this parameter
        feature_coef = [[] for _ in range(n_classes)]
        C_feature_coef[C_idx] = feature_coef
        conf_matrix_list_of_arrays = []
        scores = []

        # set feature in the feature selection element of the pipeline
        pipelineObj.set_params(**{C_param[0]: C_val})

        # run CV pipeline
        for train_index, test_index in skf.split(X, y):

            # Grab training data and test data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            pipelineObj.fit(X_train, y_train)

            # Store the list of selected features in a matrix which preserves the original size (1, 0s)
            C_feature_sel[C_idx].append(pipelineObj['selectfrommodel'].get_support())

            # Extract the coefficients the model generated, create a similar matrix, but (coef, 0s)
            # Since a model is fit for each drug, must iterate across drugs
            for drug_idx in range(n_classes):
                C_feature_coef[C_idx][drug_idx].append(pipelineObj['logisticregression'].coef_[drug_idx])

            # For ease of plotting later, just save the coefficients, as its already formated in feature * class
            C_coef_mat.append(pipelineObj['logisticregression'].coef_)

            # Run the model on the test data
            x_test_predict = pipelineObj.predict(X_test)

            # Extract the confusion matrix for the split
            conf_matrix = confusion_matrix(y_test, x_test_predict)

            # Append the confusion matrix to the larger stack
            conf_matrix_list_of_arrays.append(conf_matrix)

            # Extract the score, along the diagonal.
            scores.append(np.mean(np.diag(conf_matrix)))

        conf_matrix_list_of_arrays_mod = np.array(conf_matrix_list_of_arrays)

        C_scores[C_idx] = scores

        # Plot Confusion Matrix - # CHECK TITLES
        titleStr = str(pipelineObj['logisticregression']) + ' Feature Sel via L1, C = ' + str(C_val)
        # pf.plotConfusionMatrix(scores, drugNames,conf_matrix_list_of_arrays, '', titleStr, dirDict)

        # Prepare the confusion matrix plot
        mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays_mod, axis=0)
        C_conf_mat[C_idx] = mean_of_conf_matrix_arrays

    # Process score data for creating a line plot
    if 0:
        C_mean_score = [np.mean(x) for x in C_scores]
        C_mean_std = [np.std(x) for x in C_scores]

        C_vec_transformed = np.log10(C_vec)

        plt.errorbar(C_vec_transformed, C_mean_score, yerr=C_mean_std, fmt='o-', capsize=5)
        plt.gca().set_xticks(C_vec_transformed)
        plt.gca().set_xticklabels([f"{v:.1f}" for v in C_vec_transformed])
        # plt.gca().set_xscale('log')

        plt.xlabel('C Regularization (log scale)')
        plt.ylabel('Accuracy (SD)')
        plt.title('Accuracy vs Inverse Regularization Strength')
        plt.show()

    # Reformat the stored coefficient data into a sns heatmap - make as large as needed
    FeatureCount = []
    for C_score_set in C_feature_sel:
        FeatureCount.append([sum(x) for x in C_score_set])

    # Create the matrix of coefficients and 0s
    drug_median_coef = [[] for _ in range(len(drugNames))]
    
    for drug_idx in range(len(drugNames)):
        # For each drug, we want to make a n_features by C_vec table, where 0 is present if the coefficient is never used, and 0 - 1 is present for features which were used in at least one split.

        coef_zero_mat = []
        coef_zero_split_mat = []
        coef_zero_median = []

        for C_idx in range(len(C_vec)):
            # Create this temporary variable to create a mean coefficient across splits
            coef_per_split = []

            # Extract data
            for split_idx in range(len(C_feature_sel[C_idx])):
                booleanArray = C_feature_sel[C_idx][split_idx]
                coefData = C_feature_coef[C_idx][drug_idx][split_idx]
                coef_zero_vec = hf.replace_ones_with_integers(booleanArray, coefData)
                coef_per_split.append(coef_zero_vec)

            # For retrieve the drug data across all the
            coef_per_split = np.array(coef_per_split)
            coef_per_split = coef_per_split.transpose()
            coef_mean = np.mean(coef_per_split, axis=1)
            coef_mean = coef_mean.reshape(-1, 1)

            # Append the coef_mean, which is the mean of 0s and 1s present across all the splits, to this larger matrix
            coef_zero_mat.append(coef_mean)
            coef_zero_split_mat.append(coef_per_split)  # Store the coefficients
            # Take the median value across all CVs and append here
            coef_zero_median.append(np.median(coef_per_split, axis=1))

        # Reformat matrix to allow for plotting of heatmap.
        coef_zero_mat = np.array(coef_zero_mat)
        coef_zero_mat = np.squeeze(coef_zero_mat)
        coef_zero_mat = coef_zero_mat.transpose()
        coef_zero_split_mat = np.array(coef_zero_split_mat)
        
        # format other collected stuff for other plot
        # Process the median values for each coeficient in each C value
        coef_zero_median = np.array(coef_zero_median)
        coef_zero_median = coef_zero_median.transpose()
        drug_median_coef[drug_idx] = coef_zero_median

        # Create heatmap
        # create_heatmaps_allC(coef_zero_split_mat, 0, f'{drugNames[drug_idx]} Coefficients Per Split, C = ', C_vec, dirDict)
        print(f'Done {drugNames[drug_idx]}')

    # plottingData[drug_idx][features][C_idx]
    plottingData = np.array(drug_median_coef)
    # create_heatmaps_perDrug(plottingData, 'Per Drug Median Coefficients across Splits', drugNames, 'C', C_vec, dirDict)
    plottingData = np.swapaxes(plottingData, 0, 2)
    # create_heatmaps_perDrug(plottingData, 'Per C Median Coefficients across Splits', [f"C = {x}" for x in C_vec], 'Drug', drugNames, dirDict)
    
    # Pull all the features for each value
    C_features = [[] for _ in range(len(C_vec))]

    for C_idx in range(len(C_vec)):

        drug_features = [[] for _ in range(len(drugNames))]

        for drug_idx in range(len(drugNames)):
            drug_C_coeffs = plottingData[C_idx, :, drug_idx]

            # Scatter Plot
            keepInd = np.round(drug_C_coeffs,2) != 0
            featureNamesPlot = featureNames[keepInd]
            drug_C_coeffs = drug_C_coeffs[keepInd]

            # create a sorting index
            sorted_indices = sorted(range(len(drug_C_coeffs)), key=lambda i: abs(drug_C_coeffs[i]), reverse=True)

            # Store the sorted features, largest coefficients to smallest
            drug_coeff_sorted = drug_C_coeffs[sorted_indices]
            feature_names_sorted = featureNamesPlot[sorted_indices]
            # Combine arrays into a single list
            drug_features[drug_idx] = [(np.round(num,2), string) for num, string in zip(drug_coeff_sorted, feature_names_sorted)]

            # Plotting things
            if 0:
                sns.scatterplot(drug_C_coeffs)
                plt.axhline(0, color='black')
                threshold = np.percentile(np.abs(drug_C_coeffs), 90)
                plt.axhline(threshold, color='red', linestyle='--')
                plt.axhline(-threshold, color='red', linestyle='--')

                # Put the name of the features with the largest values up
                for i in range(len(drug_C_coeffs)):
                    if np.abs(drug_C_coeffs[i]) > threshold:
                        plt.text(i, drug_C_coeffs[i], f'{featureNamesPlot[i]}', color='red')

                plt.xlabel('Feature Index')
                plt.ylabel('Coefficient')
                plt.title(f'Coefficient weights for {drugNames[drug_idx]} at C = {C_vec[C_idx]} (90th percentile = {threshold:.2f})')
                plt.show()

        C_features[C_idx] = drug_features


    drug_C_mat = np.array(C_features)
    for drug_idx in np.arange(0, len(drugNames)):
        for C_idx in np.arange(0, len(C_vec)):
            combined_list = drug_C_mat[C_idx, drug_idx]

            # Extract absolute values and signs
            abs_values = [abs(num) for num, _ in combined_list]
            signs = ['blue' if num >= 0 else 'red' for num, _ in combined_list]

            # Get strings for left labels
            strings = [string for _, string in combined_list]

            # Create a horizontal bar plot
            plt.figure(figsize=(10, len(strings)/8))
            plt.barh(strings, abs_values, color=signs)

            # Add labels and title
            plt.yticks(fontsize=8)
            plt.margins(y=0)
            plt.xlabel('Coefficient Magnitude')
            plt.ylabel('Features')
            plt.title(f"{drugNames[drug_idx]} at C = {C_vec[C_idx]}:")

            plt.savefig(f'{dirDict["classifyDir"]}/{drugNames[drug_idx]}_C_{C_vec[C_idx]}.png', bbox_inches='tight')

            # Color code y-axis labels
            # for i, string in enumerate(strings):
            #     plt.gca().get_yticklabels()[i].set_color(signs[i])

            # Show the plot
            plt.show()

def emptyFxn(X):
    return X