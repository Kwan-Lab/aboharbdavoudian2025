import sys, os, shap
import pandas as pd
import numpy as np
import pickle as pkl

# from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import helperFunctions as hf
import plotFunctions as pf

from copy import deepcopy

from sklearn import preprocessing, linear_model, svm
from sklearn.ensemble import RandomForestClassifier
# Used to create a consistent processing pipeline prior to training/testing.
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SequentialFeatureSelector, SelectFdr, SelectFwe
from sklearn.feature_selection._univariate_selection import _BaseFilter

from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# For the custom feature selection module below
from numbers import Real
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from mrmr import mrmr_classif
from boruta import BorutaPy

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

def bootstrap_fstat(pandasdf, classifyDict, dirDict):
        
    # Reformat the data into a format for sampling
    dirDict = hf.dataStrPathGen(classifyDict, dirDict)

    X, y, featureNames, labelDict = reformat_pandasdf(pandasdf, classifyDict, dirDict)

    # Make a folder for images
    outDirPath = os.path.join(dirDict['outDir'], 'Fstat')
    if not os.path.exists(outDirPath):
        os.mkdir(outDirPath)

    pandasdf_labeled = pd.DataFrame(np.hstack([y.reshape(-1, 1), X]), columns=np.append('label', featureNames))
    
    feature_scores = []
    for idx in np.arange(1e3):
        pdf_sample = pandasdf_labeled.sample(frac=.75, replace=True)
        f_stat, p_val = f_classif(pdf_sample.drop(['label'], axis=1), pdf_sample['label'])
        feature_scores.append(f_stat)

    feature_scores = np.array(feature_scores)

    feature_scores_df = pd.DataFrame(feature_scores, columns=featureNames)

    column_means = np.round(feature_scores_df.median(), 2)
    sorted_columns = column_means.sort_values(ascending=False)
    sorted_df = feature_scores_df[sorted_columns.index]
    new_col_names = [f"{x} ({num})" for x, num in zip(sorted_df.columns, sorted_columns.values)]
    new_col_dict = dict(zip(sorted_df.columns, new_col_names))

    # Create the same plots for some regions of interest.
    regionSets = ['Top', 'Control']

    for regionL in regionSets:

        if regionL == 'Top':
            if classifyDict['label'] == 'drug':
                topVar = 10
                num_pieces = 1
            else:
                topVar = 40
                num_pieces = 4

            sorted_df_top = sorted_df.iloc[:, 0:topVar-1]
            piece_size = int(np.ceil(len(sorted_df_top.columns) / num_pieces))
            fontSizePlot = 14

        else:
            regionList = ['ACAd', 'ACAv', 'ILA', 'PL', 'MOs']
            sorted_df_top = sorted_df.loc[:,sorted_df.columns.isin(regionList)]
            num_pieces = 1
            fontSizePlot = 8

        sorted_df_top = sorted_df_top.rename(columns=new_col_dict)

        fig = plt.figure(figsize=(num_pieces * 4, 5))

        for idx in np.arange(num_pieces):
            start_col = idx * piece_size
            end_col = start_col + piece_size

            data_slice = sorted_df_top.iloc[:, start_col:end_col]
            plt.subplot(1, num_pieces, idx+1)
            sns.violinplot(data=data_slice, orient='h')
            plt.xlim([np.nanpercentile(data_slice.values, q=5), np.nanpercentile(data_slice.values, q=95)])

        fig.suptitle(f" {regionL} Regions {classifyDict['data']} ranked by F stat on {classifyDict['label']}", fontsize=fontSizePlot, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outDirPath, f"{classifyDict['data']}_{classifyDict['label']}_{regionL}.png"), format='png', bbox_inches='tight')
        plt.show()

def classifySamples(pandasdf, classifyDict, plotDict, dirDict):

    # ================== Data and Pipeline Building ==================

    # Generate a directory for the data upon formating. 
    dirDict = hf.dataStrPathGen(classifyDict, dirDict)

    # Reformat the data into a format which can be used by the classifiers.
    X, y, y_dataset, featureNames, labelDict = reformat_pandasdf(pandasdf, classifyDict, dirDict)

    # Use the classify dict to specify all the models to be tested.
    modelList, cvFxn, paramGrid = build_pipeline(classifyDict)

    # ================== Classification ==================

    fits = ['Real']
    if classifyDict['shuffle']:
        fits = fits + ['Shuffle']

    nestedCVSwitch = classifyDict['gridCV']
    
    YtickLabs = list(labelDict.keys())
    n_classes = len(YtickLabs)

    for fit in fits:
        if fit == 'Shuffle':
            y = shuffle(y)

        cvFxn.get_n_splits(X, y) 

        # Create a label binarizer
        y_bin = preprocessing.label_binarize(y, classes=YtickLabs)            
        
        # If the vector is already binary, expand it for formatting consistency.
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([np.abs(y_bin-1), y_bin])

        for clf in modelList:

            # Create a strings to represent the model
            modelStr, saveStr, dirDict = hf.modelStrPathGen(clf, dirDict, cvFxn.n_splits, fit)
            saveFilePath = dirDict['tempDir_outdata']

            # Check if data is already there
            if classifyDict['saveLoadswitch'] & os.path.exists(saveFilePath):
                print(f"loading model: {modelStr}")
                with open(saveFilePath, 'rb') as f:                 # Unpickle the data:
                    [classifyDict, modelList, modelStr, saveStr, featureSelSwitch, y_real, y_prob, conf_matrix_list_of_arrays, X_test_trans_list, scores, selected_features_list, selected_features_params, baseline_val, shap_values_list] = pkl.load(f)
                
            else: 
                print(f"evaluating model: {modelStr}")

                if hasattr(clf['classif'], 'penalty'):
                    penaltyStr = clf['classif'].penalty
                else:
                    penaltyStr = None
                    
                # Create a switch to determine if some type of featureSelection is taking place.
                featureSelSwitch = False
                if fit != 'Shuffle' and ('featureSel' in clf.named_steps.keys() or penaltyStr not in ('l2', None)):
                    featureSelSwitch = True

                # Initialize vectors for storing the features used to make the classification for each fold.
                y_real, y_prob, conf_matrix_list_of_arrays, X_test_trans_list, explainers, scores = [], [], [], [], [], []
                selected_features_list, selected_features_params = [[] for _ in range(n_classes)], [[] for _ in range(n_classes)]
                
                if n_classes == 2:
                    baseline_val, shap_values_list = [], []
                else:
                    baseline_val, shap_values_list = [[] for _ in range(n_classes)], [[] for _ in range(n_classes)]

                print(f"Performing CV split ", end='')
                for idx, (train_index, test_index) in enumerate(cvFxn.split(X, y)):
                    print(f"{idx + 1} ", end='')

                    # Grab training data and test data
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    y_bin_test = y_bin[test_index, :]

                    X_train = pd.DataFrame(X_train,columns=featureNames)
                    X_test = pd.DataFrame(X_test,columns=featureNames)

                    if nestedCVSwitch and paramGrid:
                        # Do a grid search for the relevant parameters
                        grid_search = GridSearchCV(clf, paramGrid, cv=classifyDict['innerFold'], scoring='neg_log_loss', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        clf.set_params(**best_params)

                        # Report out on those models
                        print({k: v for k, v in grid_search.best_params_.items()})

                    # Fit the model
                    try:
                        clf.fit(X_train, y_train)
                    except:
                        print(f"\n Failed to fit CV {idx} - {modelStr}")
                        print(f"\n Next Idx")
                        continue

                    # Create some structures to interpret the results
                    if 'featureSel' in clf.named_steps.keys():
                        feature_selected = featureNames[clf['featureSel'].get_support()]
                    else:
                        feature_selected = featureNames

                    # SHAP Related code
                    X_test_trans = pd.DataFrame(clf[:-1].transform(X_test), columns=feature_selected, index=test_index)
                    if n_classes == 2:
                        explainer = shap.LinearExplainer(clf._final_estimator, X_test_trans, feature_perturbation=classifyDict['featurePert'])
                    else:
                        # Multiclass explainer must use interventional perturbation
                        explainer = shap.LinearExplainer(clf._final_estimator, X_test_trans, feature_perturbation='interventional')
                        
                    explainers.append(explainer)
                    explain_shap_vals = explainer.shap_values(X_test_trans)

                    if n_classes == 2:
                        shap_values_test = pd.DataFrame(explain_shap_vals, columns=feature_selected, index=test_index)
                        shap_values_list.append(shap_values_test.reset_index())
                        baseline_val.append(explainer.expected_value)
                    else:
                        shap_values_test = [pd.DataFrame(x, columns=feature_selected, index=test_index) for x in explain_shap_vals]
                        for idx, shap_val in enumerate(shap_values_test):
                            shap_values_list[idx].append(shap_val.reset_index(drop=True))
                            baseline_val[idx].append(explainer.expected_value)

                    # Store the results
                    X_test_trans_list.append(X_test_trans.reset_index())

                    # if best K, get ride of the non-label names
                    if featureSelSwitch:
                        featureNamesSub = featureNames[clf['featureSel'].get_support(indices=True)]
                    else:
                        featureNamesSub = featureNames

                    # Predict answers for the X_test set
                    x_test_predict = clf.predict(X_test)

                    # Extract the confusion matrix for the split
                    if classifyDict['label'] == 'drug':
                        conf_matrix = confusion_matrix(y_test, x_test_predict, labels=YtickLabs) #, labels=YtickLabs
                    else:
                        conf_matrix = confusion_matrix(y_test, x_test_predict)

                    # In cases where test set has more than 1 instance of a class, normalize reach row.
                    if np.max(np.sum(conf_matrix, axis=1)) != 1:
                        sums = np.sum(conf_matrix, axis=1)
                        conf_matrix = conf_matrix / sums[:, np.newaxis]

                    # Append the confusion matrix to the larger stack
                    conf_matrix_list_of_arrays.append(conf_matrix)
                    
                    # Extract the score, along the diagonal.
                    scores.append(np.mean(np.diag(conf_matrix)))

                    # Calculate probabilities for PR Curve - Prior to storing, resort (clf class order can't be changed)
                    y_scores = clf.predict_proba(X_test)

                    if not np.all(clf.classes_ == YtickLabs):
                        original_list = YtickLabs
                        target_list = clf.classes_
                        mapping = {element: i for i, element in enumerate(target_list)}
                        index = [mapping[element] for element in original_list]
                        y_prob.append(y_scores[:, index])
                    else:
                        y_prob.append(y_scores)
                        
                    y_real.append(y_bin_test)

                    if featureSelSwitch:

                        # if featureSel done as module, final_estimator will only have info on selected features.
                        if 'featureSel' in clf.named_steps.keys():
                            featureNamesSub = featureNames[clf['featureSel'].get_support(indices=True)]
                        else:
                            featureNamesSub = featureNames

                        # If featureSel done as part of estimator, this can be determined based on coefs.
                        if penaltyStr not in ('l2', None):
                            bool_array = clf['classif'].coef_ != 0
                            featureNamesSub = featureNamesSub[bool_array.flatten()]

                        # Append the selected features across splits. - Consider removing, as features are the same across classes
                        for idx in np.arange(n_classes):
                            selected_features_list[idx].append(featureNamesSub)
                            if nestedCVSwitch:
                                selected_features_params[idx].append(best_params)
                            else:
                                selected_features_params[idx].append(dict())

                # End of CV Split

                if classifyDict['saveLoadswitch']:
                    # save all the products
                    saveList = [classifyDict, modelList, modelStr, saveStr, featureSelSwitch, y_real, y_prob, conf_matrix_list_of_arrays, X_test_trans_list,
                                scores, selected_features_list, selected_features_params, baseline_val, shap_values_list]
                    with open(saveFilePath, 'wb') as f:
                        pkl.dump(saveList, f)

            ## Plotting code for the fit model and compiled data
            # ================== Plotting ==================
            # PR Curve - save the results in a dict for compiling into a bar plot.
            auc_dict = pf.plotPRcurve(n_classes, y_real, y_prob, labelDict, modelStr, plotDict['plot_PRcurve'], fit, dirDict)

            # Create a structure which saves results for plotting elsewhere.
            score_dict = dict()
            score_dict['auc'] = auc_dict
            score_dict['scores'] = scores
            score_dict['featuresPerModel'] = selected_features_list[0]
            score_dict['compLabel'] = ' vs '.join(labelDict.keys())

            # Save
            dictPath = os.path.join(dirDict['outDir_model'], f'scoreDict_{fit}.pkl')
            with open(dictPath, 'wb') as f:
                pkl.dump(score_dict, f)

            # # Shape data into a table for correlation
            if plotDict['featureCorralogram']:
                featureCountLists = hf.feature_model_count(selected_features_list[0])
                feature_df = pd.DataFrame(X, columns=featureNames, index=y_dataset)
                pf.correlation_subset(feature_df, pandasdf, featureCountLists, plotDict['shapSummaryThres'], classifyDict, dirDict)
        
            # SHAP
            if fit != 'Shuffle':
                if plotDict['plot_SHAPforce']:
                    pf.plot_shap_force(X_test_trans_list, shap_values_list, baseline_val, y_real, labelDict, plotDict, dirDict)

                if plotDict['plot_SHAPsummary']:

                    pf.plot_shap_summary(X_test_trans_list, shap_values_list, y, n_classes, plotDict, classifyDict, dirDict)
                    # pf.plot_shap_bar(explainers, X_test_trans_list, shap_values_list, y, n_classes, plotDict, classifyDict, dirDict)

            # Confusion Matrix
            if plotDict['plot_ConfusionMatrix']:
                pf.plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, saveStr, dirDict)

            # Report out feature selected if
            if featureSelSwitch:
                hf.stringReportOut(selected_features_list[0], selected_features_params, YtickLabs, dirDict)

            # if fit != 'Shuffle' and len(labelDict) != 2:
            #     findConfusionMatrix_LeaveOut(daObj, clf, X, y, labelDict, 8, fit=fit)

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

def build_pipeline(classifyDict):
    # a function which accepts 'classifyDict' with fields described below, and returns a list of pipeline object(s) and a paramGrid for use in gridSearchCV

    max_iter = classifyDict['max_iter']
    paramGrid = dict()
    pipelineList = []

    # Create a model with each of the desired feature counts.
    if 'LogReg' in classifyDict['model'] or 'SVC' in classifyDict['model']:
        paramGrid['classif__C'] = classifyDict['pGrid']['classif__C']

    # module for feature transformation:
    if classifyDict['model_featureTransform']:
        transMod = PowerTransformer(method='yeo-johnson', standardize=False)

    # Select Feature scaling model
    if classifyDict['model_featureScale']:
        scaleMod = RobustScaler()

    # Select Feature selection model
    match classifyDict['model_featureSel']:
        case 'Boruta':
            featureSelMod = BorutaFeatureSelector(feature_sel='all') # Can swap for 'feature_sel'='strong' to only include strong rec's from Boruta.
        case 'MRMR':
            # featureSelMod = FunctionTransformer(MRMRFeatureSelector, kw_args={'n_features_to_select': 10, 'pass_y': True, 'featureSel__feature_names_out': True}, )
            featureSelMod = MRMRFeatureSelector(n_features_to_select=30)
            modVar = 'n_features_to_select'
        case 'Univar':
            featureSelMod = SelectKBest(score_func=f_classif, k='all')
            modVar = 'k'
        case 'Fdr':
            featureSelMod = SelectFdr(alpha=classifyDict['model_featureSel_alpha'])
        case 'Fwe':
            featureSelMod = SelectFwe(alpha=classifyDict['model_featureSel_alpha'])
        case 'Fwe_BH':
            featureSelMod = SelectFwe_Holms(alpha=classifyDict['model_featureSel_alpha'])
        case 'mutInfo':
            featureSelMod = SelectKBest(score_func=mutual_info_classif, k='all')
            modVar = 'k'
        case 'RFE':
            featureSelMod = SequentialFeatureSelector(classif, n_features_to_select=10, direction="forward")
            modVar = 'n_features_to_select'

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
        case 'svm': # Does not play well with current SHAP implementation. Would require shap.KernelExplainer instead, which has a different formatting for outputs.
            classif = svm.SVC(C=1, kernel='linear', max_iter=max_iter, probability=True)
            # paramGrid['classif__l1_ratio'] = classifyDict['pGrid']['classif__l1_ratio']
        case _:
            ValueError('Invalid modelStr')

    if 'transMod' in locals():
        pipelineList.append(('featureTrans', transMod))

    if 'scaleMod' in locals():
        pipelineList.append(('featureScale', scaleMod))

    # Create the pipeline, first to last
    if 'featureSelMod' in locals():
        pipelineList.append(('featureSel', featureSelMod))

    pipelineList.append(('classif', classif))

    # Create a base model
    clf = Pipeline(pipelineList)
    
    if classifyDict.get('model_featureSel') not in ('None', 'Fdr', 'Fwe', 'Fwe_BH') and classifyDict['model_featureSel_mode'] == 'gridCV':
        paramGrid['featureSel__k'] = classifyDict['model_featureSel_k']

    modelList = []
    # Iterate over the desired feature counts
    if classifyDict.get('model_featureSel') not in ('None', 'Fdr', 'Fwe', 'Fwe_BH', 'Boruta') and classifyDict['model_featureSel_mode'] == 'modelPer':
        for k in classifyDict['model_featureSel_k']:
            clf_copy = deepcopy(clf)
            modelList.append(clf_copy.set_params(**{f"featureSel__{modVar}": k}))
    else:
        modelList.append(clf)  

    # ================== Generate and pass cross validation module ==================

    if classifyDict['CVstrat'] == 'StratKFold':
        cvFxn = StratifiedKFold(n_splits=classifyDict['CV_count'])  # 8 examples of each label
    elif classifyDict['CVstrat'] == 'ShuffleSplit':
        cvFxn = StratifiedShuffleSplit(n_splits=classifyDict['CV_count'], test_size=classifyDict['test_size'])
    
    return modelList, cvFxn, paramGrid

def reformat_pandasdf(pandasdf, classifyDict, dirDict):
    # Function which performs steps related to feature filtering, feature aggregation, and data formatting
    # to be passed forward for classification.

    conv_dict = hf.create_drugClass_dict(classifyDict)

    # ================== Perform Filtration and agglomeration as desired ==================
    data_param_string = dirDict['data_param_string']
    filtAggFileName = os.path.join(dirDict['tempDir_data'], "data_preprocessed.pkl")

    # If the data labels are not for 'drug', filter
    if classifyDict['label'] != 'drug' and bool(conv_dict):
        pandasdf[classifyDict['label']] = pandasdf['drug'].map(conv_dict)
        pandasdf = pandasdf.dropna(subset=[classifyDict['label']])

    if (classifyDict['featurefilt'] or classifyDict['featureAgg']):

        if not os.path.exists(filtAggFileName):

            print(f"Generating filtered and aggregated data file... {data_param_string}")
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
            print(f"Loading filtered and aggregated data from file...{data_param_string}")
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
        
    # ================== Sort the data for classification ==================
    if classifyDict['label'] == 'drug':
        datasetNames = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']
        new_list = [f'{item}{i}' for item in datasetNames for i in range(1, 9)]
        ls_data_agg = ls_data_agg.reindex(new_list)

    # ================== Shape data for classification ==================

    X = np.array(ls_data_agg.values)
    y = np.array([x[0:-1] for x in np.array(ls_data_agg.index)])
    y_dataset = ls_data_agg.index

    # Since these labels come from the dataset, not the drug, they need to be converted to the appropriate label.
    y = ['6-F-DET' if item == '6FDET' else item for item in y]
    y = ['5MEO' if item == 'DMT' else item for item in y]
    y = np.array(y)

    # If the data labels are not for 'drug', convert to appropriate labels
    if bool(conv_dict):
        # Use the conv_dict to convert the labels, filter the data
        y = np.array([conv_dict.get(item, np.nan) for item in y])
        labels = np.unique(y)
    else:
        customOrder = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']
        y = pd.Categorical(y, categories=customOrder, ordered=True)
        labels = y.unique()

    y_Int_dict = dict(zip(labels, range(0, len(labels))))
    y = np.array(y)

    # Create a list of the features
    featureNames = np.array(ls_data_agg.columns.tolist())

    return X, y, y_dataset, featureNames, y_Int_dict

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

class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_sel='all'):
        self.feature_sel = feature_sel
    
    def fit(self, X, y=None):

        boruta = BorutaPy(estimator = RandomForestClassifier(), n_estimators = 'auto', max_iter = 100, perc=95, verbose=0)
        # Perc - percentile for real feature importance compared to distribution of shadow features. 100 believed to be too strict.
        
        ### fit Boruta (it accepts np.array, not pd.DataFrame)
        boruta.fit(np.array(X), np.array(y))

        # store features selected in self for later transformation
        if self.feature_sel == 'all':
            self.selected_features_ = boruta.support_ | boruta.support_weak_
        elif self.feature_sel == 'strong':
            self.selected_features_ = boruta.support_

        return self
    
    def get_support(self, indices=True):
        return self.selected_features_
    
    def transform(self, X, y=None):
        # Pick out previously selected variables
        # feature_to_use = self.selected_features_[0:self.n_features_to_select]
        if type(X) == pd.DataFrame:
            X_fit = X.iloc[:, self.selected_features_]
        else:
            X_fit = X[:, self.selected_features_]
        # print(f"transformed X: {X_fit.shape}")

        return X_fit

class SelectFwe_Holms(_BaseFilter):
    # Modified version of SelectFwe found in sklearn which performs the Bonferroni-Hocberg correction

    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "alpha": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        # calculate cut off per p-value
        m = len(self.pvalues_)
        j = np.arange(1, m + 1)
        denominatorVec = np.array(m + 1 - j)
        threshold = self.alpha/denominatorVec

        # Sort p-values
        sorted_indices = np.argsort(self.pvalues_)
        reverse_indices = np.argsort(sorted_indices)
        sorted_pvalues = self.pvalues_[sorted_indices]

        # Find which are below the threshold
        below_threshold = sorted_pvalues < threshold
        support_mask = below_threshold[reverse_indices]

        return support_mask
