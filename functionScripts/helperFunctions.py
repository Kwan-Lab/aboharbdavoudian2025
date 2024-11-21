import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import os, sys
import pickle as pkl
from collections import defaultdict, Counter

def create_drugClass_dict(classifyDict):
    # Create a dictionary to convert drug names to drug classes
    # Ideally switch this to a match case, but if statements here for a bit more backwards compatability.
    conv_dict = dict()

    # Drug Class vs Drug Class
    if classifyDict['label'] == 'class_5HT2A':
        # 5-HT2A agonist psychedelics (Psilo, 5MEO) vs entactogen (MDMA)
        conv_dict['PSI'] = 'PSI/5MEO'
        conv_dict['5MEO'] = 'PSI/5MEO'
        conv_dict['MDMA'] = 'MDMA'
    if classifyDict['label'] == 'class_5HTR':
        # Typtamines (Psilo, 5MEO) vs non-Hallucinogenic trypamines (6-F-DET)
        conv_dict['PSI'] = 'PSI/5MEO'
        conv_dict['5MEO'] = 'PSI/5MEO'
        conv_dict['6-F-DET'] = '6-F-DET'
    if classifyDict['label'] == 'class_Psy_NMDA':
        # Fast Psychedelic vs Fast NMDA-R Agonist (Psi, 5MEO vs Ketamine)
        conv_dict['PSI'] = 'PSI/5MEO'
        conv_dict['5MEO'] = 'PSI/5MEO'
        conv_dict['KET'] = 'KET'

    # Drug Class vs Drug
    if classifyDict['label'] == 'class_PsiKet':
        conv_dict['PSI'] = 'PSI'
        conv_dict['KET'] = 'KET'
    if classifyDict['label'] == 'class_Psi5MEO':
        conv_dict['PSI'] = 'PSI'
        conv_dict['5MEO'] = '5MEO'
    if classifyDict['label'] == 'class_PsiMDMA':
        conv_dict['PSI'] = 'PSI'
        conv_dict['MDMA'] = 'MDMA'
    if classifyDict['label'] == 'class_PsiSSRI':
        conv_dict['PSI'] = 'PSI'
        conv_dict['A-SSRI'] = 'A-SSRI'
    if classifyDict['label'] == 'class_Trypt':
        # 5-HT2A favored (Psilo) vs 5-HT1A favored (5MEO)
        conv_dict['PSI'] = 'PSI'
        conv_dict['5MEO'] = '5MEO'
    if classifyDict['label'] == 'class_DT':
        conv_dict['5MEO'] = '5MEO'
        conv_dict['6-F-DET'] = '6-F-DET'
    if classifyDict['label'] == 'class_PsiSSRI':
        conv_dict['PSI'] = 'PSI'
        conv_dict['A-SSRI'] = 'A-SSRI'
    if classifyDict['label'] == 'class_SSRI':
        conv_dict['A-SSRI'] = 'A-SSRI'
        conv_dict['C-SSRI'] = 'C-SSRI'
    if classifyDict['label'] == 'class_PsiDF':
        conv_dict['PSI'] = 'PSI'
        conv_dict['6-F-DET'] = '6-F-DET'

    # Leave Out analyses - training/testing set disparities.
    # Include all relevant classes here
    # Classes below represent All the data passed forward - Testing Set
    # the classes left out of the training data are defined in classifyDict['LO_drug'] (Training Set = Below Labels - classifyDict['LO_drug'])
    if classifyDict['label'] == 'LO_6FDET':
        conv_dict['PSI'] = 'PSI'
        conv_dict['KET'] = 'KET'
        conv_dict['5MEO'] = '5MEO'
        conv_dict['MDMA'] = 'MDMA'
        conv_dict['A-SSRI'] = 'A-SSRI'
        conv_dict['C-SSRI'] = 'C-SSRI'
        conv_dict['SAL'] = 'SAL'
        conv_dict['6-F-DET'] = '6-F-DET'
    if classifyDict['label'] == 'LO_6FDET_SSRI':
        conv_dict['PSI'] = 'PSI'
        conv_dict['KET'] = 'KET'
        conv_dict['5MEO'] = '5MEO'
        conv_dict['MDMA'] = 'MDMA'
        conv_dict['A-SSRI'] = 'A-SSRI'
        conv_dict['C-SSRI'] = 'C-SSRI'
        conv_dict['SAL'] = 'SAL'
        conv_dict['6-F-DET'] = '6-F-DET'
    if classifyDict['label'] == 'LO_SSRI':
        conv_dict['PSI'] = 'PSI'
        conv_dict['KET'] = 'KET'
        conv_dict['5MEO'] = '5MEO'
        conv_dict['MDMA'] = 'MDMA'
        conv_dict['A-SSRI'] = 'A-SSRI'
        conv_dict['C-SSRI'] = 'C-SSRI'
        conv_dict['SAL'] = 'SAL'
        conv_dict['6-F-DET'] = '6-F-DET'
    # if classifyDict['label'] == 'LO_all_nSSRI':
    #     conv_dict['PSI'] = 'PSI'
    #     conv_dict['KET'] = 'KET'
    #     conv_dict['5MEO'] = '5MEO'
    #     conv_dict['MDMA'] = 'MDMA'
    #     conv_dict['SAL'] = 'SAL'
    #     conv_dict['6-F-DET'] = '6-F-DET'
        
    if not bool(conv_dict) and classifyDict['label'] != 'drug':
        raise KeyError('No dictionary found for this classification type. Check the label in classify dict is in helperFunctions.create_drugClass_dict')

    return conv_dict

def create_color_dict(dictType='drug', rgbSwitch=0, alpha_value=1, scaleVal=False):
    # Create a dictionary of colors for brain regions or drugs
    color_dict = dict()

    if dictType == 'drug':
        color_dict['PSI'] = '#228833'
        color_dict['KET'] = '#AA3377'
        color_dict['5MEO'] = '#4477AA'
        color_dict['6-F-DET'] = '#66CCEE'
        color_dict['MDMA'] = '#CCBB44'
        color_dict['A-SSRI'] = '#CC3311'
        color_dict['C-SSRI'] = '#EE6677'
        color_dict['SAL'] = '#BBBBBB'

        # Drug Combos and non-trad names.
        color_dict['PSI/5MEO'] = '#228833'
        color_dict['PSI + 5MEO'] = '#228833'
        color_dict['HTrypt'] = '#228833'
        color_dict['Non Halluc Trypt'] = '#66CCEE'
        color_dict['Ag_5HT2A'] = '#228833'
        color_dict['Entactogen'] = '#AA3377'

    elif dictType == 'brainArea':
        color_dict['Olfactory'] = '#377eb8'
        color_dict['Cortex'] = '#ff7f00'
        color_dict['Hippo'] = '#4daf4a'
        color_dict['StriatumPallidum'] = '#f781bf'
        color_dict['Thalamus'] = '#a65628'
        color_dict['Hypothalamus'] = '#984ea3'
        color_dict['MidHindMedulla'] = '#999999'
        color_dict['Cerebellum'] = '#e41a1c'

    if rgbSwitch:
        # If rgbSwitch is on, replace the entries in color_dict with RGBA values.
        for color in color_dict.items():
            color_dict[color[0]] = (int(color[1][1:3], 16), int(color[1][3:5], 16), int(color[1][5:7], 16))

        if scaleVal:
            # Scale the values between 0 and 1
            for color in color_dict.items():
                color_dict[color[0]] = tuple(ti/255 for ti in color[1])

        if alpha_value != 0:
            # addAlpha is true, add the alpha value
            for color in color_dict.items():
                color_dict[color[0]] = color[1] + (alpha_value,)  # Add alpha value to the RGB tuple

    return color_dict

def create_translation_dict(dictType='brainArea'):
    # Create a dictionary of colors for brain regions or drugs
    translation_dict = dict()

    if dictType == 'drug':

        translation_dict['PSI'] = 'Psilocybin'
        translation_dict['aPSI'] = 'Psilocybin (B1)'
        translation_dict['KET'] = 'Ketamine'
        translation_dict['aKET'] = 'Ketamine (B1)'
        translation_dict['cKET'] = 'Ketamine (B3)'
        translation_dict['5MEO'] = '5-MeO-DMT' #5-MeO-Dimethyltryptamine (5MEO)
        translation_dict['6-F-DET'] = '6-Fluoro-DET' #6-Flouro-Diethyltryptamine (6-F-DET)
        translation_dict['MDMA'] = '3,4-MDMA' 
        translation_dict['A-SSRI'] = 'Acute SSRI'
        translation_dict['C-SSRI'] = 'Chronic SSRI'
        translation_dict['SAL'] = 'Saline'

    elif dictType == 'brainArea':

        translation_dict['Olfactory'] = 'Olfactory'
        translation_dict['Cortex'] = 'Cortex'
        translation_dict['Hippo'] = 'Hippo'
        translation_dict['StriatumPallidum'] = 'Stri+Pall'
        translation_dict['Thalamus'] = 'Thalamus'
        translation_dict['Hypothalamus'] = 'Hypothalamus'
        translation_dict['MidHindMedulla'] = 'Mid Hind Medulla'
        translation_dict['Cerebellum'] = 'Cerebellum'

    elif dictType == 'classToDrug':

        translation_dict['Ag_5HT2A'] = 'PSI/5MEO'
        translation_dict['Acute SSRI'] = 'A-SSRI'
        translation_dict['Chronic SSRI'] = 'C-SSRI'
        translation_dict['Psilocybin'] = 'PSI'
        translation_dict['Ketamine'] = 'KET'
        translation_dict['H_Trypt'] = 'PSI/5MEO'
        translation_dict['Non Halluc Trypt'] = '6-F-DET'
        translation_dict['6-Fluoro-DET'] = '6-F-DET'
        translation_dict['Entactogen'] = 'MDMA'

    return translation_dict

def find_middle_occurrences(lst):

    positions = defaultdict(list)
    first_occurrences, middle_occurrences, last_occurrences, items = [], [], [], []

    # Store the positions of each unique element
    for idx, elem in enumerate(lst):
        positions[elem].append(idx)

    # Find the middle occurrence of each unique element
    for elem, pos_list in positions.items():
        middle_idx = len(pos_list) // 2
        middle_occurrence = pos_list[middle_idx]
        middle_occurrences.append(middle_occurrence)
        items.append(elem)
    
    # Return the first and last elements as well
    for item in items:
        first_occurrences.append(positions[item][0])
        last_occurrences.append(positions[item][-1])

    position_dict = dict()
    position_dict = {item: [first_occurrences[i], middle_occurrences[i], last_occurrences[i]] for i, item in enumerate(items)}

    return position_dict

def agg_cluster(lightsheet_data, classifyDict, dirDict):
    """
    Aggregates clustered data based on the provided parameters.

    Parameters:
    - lightsheet_data: The input data containing the lightsheet information.
    - classifyDict: A dictionary containing the classification parameters.
    - dirDict: A dictionary containing the directory information.

    Returns:
    - df_agged: A DataFrame containing the aggregated data.

    This function aggregates the clustered data based on the provided parameters. It takes in the lightsheet_data, which is the input data containing the lightsheet information. 
    The classifyDict parameter is a dictionary that contains the classification parameters. The dirDict parameter is a dictionary that contains the directory information.

    The aggregation can be according to 
    - Clusters the data using scipy methods for distance calculations and linkage.
    - Color codes the dendrogram labels based on brain region.
    """
    from sklearn import datasets, cluster, preprocessing, linear_model
    from scipy.spatial import distance
    from scipy.cluster import hierarchy
    import seaborn as sns

    # Variables
    # dist_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    # linkage_list = ['ward', 'average', 'average', 'average', 'average']
    cluster_count = classifyDict['cluster_count']

    # Set variable for color coding output plots
    brainAreaList= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']
    AreaIdx = dict(zip(brainAreaList, np.arange(len(brainAreaList))))
    brainAreaColorDict = create_color_dict(dictType='brainArea', rgbSwitch=0, alpha_value=1, scaleVal=False)

    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data.loc[:, colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea.loc[:, 'Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict[x] for x in regionArea.loc[:, 'Brain_Area']]
    regionArea.sort_values(by='abbreviation', inplace=True)

    # Extract data from pandas df
    df_Tilted = lightsheet_data.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])    
    X = df_Tilted.values
    
    # Plot the original data
    titleStr = f"{classifyDict['data']}"
    if 0:
        plt.figure(figsize=(40,10))
        sns.heatmap(X, square=True)
        plt.title(titleStr, fontsize=15)
        plt.show()

    else:
        X_scaled = X

    # Cluster the data using scipy methods for distance calculations and linkage
    col_row_link = hierarchy.linkage(X_scaled.T, method=classifyDict['featureSel_linkage'], metric=classifyDict['featureSel_distance'], optimal_ordering=True)
    titleStr += f", Metric: {classifyDict['featureSel_distance']}, Linkage: {classifyDict['featureSel_linkage']}"

    plt.figure(figsize=(40,10))
    dendObj = hierarchy.dendrogram(col_row_link, labels=df_Tilted.columns, leaf_rotation=90, leaf_font_size=10, color_threshold=classifyDict['cluster_thres'])
    plt.title(titleStr, fontsize=30)

    # Color code the dendrogram labels based on brain region.
    ax = plt.gca()
    for xtick in ax.get_xticklabels():
        colorCode = regionArea[regionArea[classifyDict['feature']] == xtick._text]['Region_Color'].values[0]
        xtick.set_color(colorCode)

    # Add in string below the middle leaf
    leaf_x = dendObj['leaves']
    middle_x =  ax.get_xticklabels()[round(len(leaf_x)/2)]._x
    fullList = list(brainAreaColorDict.keys())
    fullLen = len(fullList)//2
    
    for idx, (area1, area2) in enumerate(zip(fullList[:fullLen], fullList[fullLen:])):
        plt.text(middle_x*0.9, -0.1 - (idx * 0.05), area1, color=brainAreaColorDict[area1], ha='center', fontsize=20)
        plt.text(middle_x*1.1, -0.1 - (idx * 0.05), area2, color=brainAreaColorDict[area2], ha='center', fontsize=20)

    plt.axhline(y=classifyDict['cluster_thres'], color='r', linestyle='--', linewidth=3)
    plt.yticks(fontsize=30)

    figSavePath = os.path.join(dirDict['outDir_data'], 'origDendrogram.svg')
    plt.savefig(figSavePath, bbox_inches='tight', format='svg')
    plt.show()

    # Show the cluster map
    ax = sns.clustermap(X_scaled.T, row_linkage = col_row_link, col_cluster=False, square=True) #row_colors=[regionArea.Region_Color]
    plt.title(f"{titleStr}, Clustered", fontsize=15)
    plt.title(titleStr, fontsize=15)

    figSavePath = os.path.join(dirDict['outDir_data'], 'origClustered.svg')
    plt.savefig(figSavePath, bbox_inches='tight', format='svg')
    plt.show()

    # Generate new features based on clustering - average of each cluster
    new_labels = hierarchy.fcluster(col_row_link, classifyDict['cluster_thres'], criterion='distance')
    unique_values, counts = np.unique(new_labels, return_counts=True)

    df_agged = pd.DataFrame(index=df_Tilted.index)

    # Process the features which are preserved first
    single_Idx = counts == 1
    
    singleFeatureIdx = np.in1d(new_labels, unique_values[single_Idx])
    df_agged = df_agged.join(df_Tilted.loc[:, singleFeatureIdx])

    unique_values = unique_values[~single_Idx]
    counts = counts[~single_Idx]
    
    print(f'Clustering done: {len(counts)} clusters generated from {np.sum(counts)} features.')
    
    # Assign new names to features which are clustered, save long abbreviated names to a text file.
    file = open(os.path.join(dirDict['outDir_data'], 'Cluster_members.txt'), 'w')
    ClusterCount = 1
    for idx, val in enumerate(unique_values):
        featureList = list(df_Tilted.columns[new_labels == val])
        if len(featureList) < 5:
            new_feature_name = '-'.join(featureList)
        else:
            new_feature_name = f"Clus{ClusterCount}_n{len(featureList)}"
            ClusterCount += 1
            file.write(f'{new_feature_name}: {featureList} \n\n')

        df_agged[f'{new_feature_name}'] = df_Tilted[featureList].apply(lambda x: x.mean(), axis=1)

    file.close()

    plt.figure(figsize=(30,10))
    sns.heatmap(df_agged.values, square=True)
    plt.title(f"{classifyDict['data']}, Clustered into {len(np.unique(new_labels))} Features", fontsize=15)
    plt.xlabel('Feature', fontsize=15)

    # Save the figure in the dirDict['tempDir']
    figSavePath = os.path.join(dirDict['outDir_data'], 'clustered.svg')
    plt.savefig(figSavePath, bbox_inches='tight', format='svg')
    plt.show()

    # Following the aggregation, check for multicollinearity
    # from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    # vif_val = pd.DataFrame({"Col":df_agged.columns})
    # vif_val["VIF"] = [vif(df_agged.values, i) for i in range(df_agged.shape[1])]
    # vif_val
    # print(vif_val)

    return df_agged

def create_region_to_area_dict(lightsheet_data, dataFeature):
    
    # For explicit sequence, set to false and modify.
    if 1:
        brainAreas = lightsheet_data.Brain_Area.unique()
    else:
        brainAreas= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']

    # Determines index and eventual sorting of brain areas. 
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))

    if isinstance(dataFeature, str):
        dataFeature = [dataFeature]

    dataFeature.append('Brain_Area')
    regionArea = lightsheet_data.loc[:, dataFeature]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea.loc[:, 'Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)

    return regionArea

def create_brainArea_dict(dictType):
    # Create a dictionary which can aid in plotting by converting names to

    brainAreaList= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']
    if dictType == 'short':
        brainAreaListPlot= ['Olfactory', 'Cortex', 'Hippo', 'Stri+Pall', 'Thalamus', 'Hypothalamus', 'Mid Hind Medulla', 'Cerebellum']
    elif dictType == 'long':
        brainAreaListPlot= ['Olfactory', 'Cortex', 'Hippocampus', 'Striatum and Pallidum', 'Thalamus', 'Hypothalamus', 'Midbrain, Hind Brain, and Medulla', 'Cerebellum']

    brainAreaPlotDict = dict(zip(brainAreaList, brainAreaListPlot))

    return brainAreaPlotDict

def replace_strings_with_dict(input_strings, translate_dict):
    replaced_strings = []

    for string in input_strings:
        for key, value in translate_dict.items():
            string = string.replace(key, value)
        replaced_strings.append(string)

    return replaced_strings

def filter_features(pandasdf, classifyDict):
    # Takes in a pandas dataframe, returns a version where features are filtered and the remainder are aggregated.
    import re
    plotHist = 0
    # ======= Filter Phase ========
    # Filter based on percentile (get rid of very high or very low features)
    
    # Find features to remove
    if classifyDict['filtType'] == 'max':
        thres = np.percentile(pandasdf[classifyDict['data']], 99.5)
        pandasdf_over = pandasdf[pandasdf[classifyDict['data']] >= thres]
        features_remove = pandasdf_over[classifyDict['feature']].unique()
        strWord = 'over'
    if classifyDict['filtType'] == 'min':
        thres = np.percentile(pandasdf[classifyDict['data']], .5)
        pandasdf_under = pandasdf[pandasdf[classifyDict['data']] <= thres]
        features_remove = pandasdf_under[classifyDict['feature']].unique()
        strWord = 'under'
        

    # Visualize the flatten data
    if plotHist:
        df_Tilted = pandasdf.pivot(index=classifyDict['feature'], columns='dataset', values=classifyDict['data'])
        plt.hist(df_Tilted.values.flatten(), bins=100, edgecolor='black')
        x_lim = plt.xlim()
        y_lim = plt.ylim()

        plt.xlim(0, x_lim[1])
        plt.ylim(y_lim[0], 20)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram Before')

    pandasdf_filt = pandasdf[~pandasdf[classifyDict['feature']].isin(features_remove)]

    if plotHist:
        plt.hist(np.array(pandasdf_filt[classifyDict['data']]), bins=50, edgecolor='black')
        plt.show()
    
    feature_n_old = len(pandasdf[classifyDict['feature']].unique())
    feature_n_new = len(pandasdf_filt[classifyDict['feature']].unique())

    print(f"feature count shifted from {feature_n_old} to {feature_n_new}, removing all features with instance of '{classifyDict['data']}' {strWord} {thres}")

    return pandasdf_filt

def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def replace_ones_with_integers(binary_vector, integer_array):
    result = []
    integer_index = 0

    for element in binary_vector:
        if element == 1:
            if integer_index < len(integer_array):
                result.append(integer_array[integer_index])
                integer_index += 1
            else:
                raise ValueError(
                    "Not enough integers in the integer_array to replace all ones")
        else:
            result.append(0)

    return result

def check_phrase_in_keys(phrase, dictionary):
    import re

    phrase = re.escape(phrase)
    phrase = phrase.replace('\\*', '.*')
    pattern = '^' + phrase + '$'
    regex = re.compile(pattern)

    matching_keys = []

    for key in dictionary.keys():
        if regex.match(key):
            matching_keys.append(key)

    return matching_keys

def convert_to_highest_entry_array(input_arr):
    """
    Given a 2D numpy array, returns a new array where each row has only one non-zero element:
    the element with the highest value in the original row.
    """
    output_arr = np.zeros_like(input_arr)

    for i, row in enumerate(input_arr):
        max_index = np.argmax(row)
        output_arr[i, max_index] = 1

    return output_arr

def conciseStringReport(strings, counts):

    # Create a dictionary by zipping the two lists
    data_dict = dict(zip(strings, counts))

    # Create a reverse dictionary with counts as keys and list of strings as values
    reverse_dict = {}
    for key, value in data_dict.items():
        if value not in reverse_dict:
            reverse_dict[value] = []
        reverse_dict[value].append(key)

    myKeys = list(reverse_dict.keys())
    myKeys.sort()
    reverse_dict = {i: reverse_dict[i] for i in myKeys}

    # Initialize an empty string to store the result
    result_string = ""

    # Iterate over the reverse dictionary and append the key-value pairs to the result_string
    for count, string_list in reverse_dict.items():
        string_group = ', '.join(string_list)
        result_string += f"Present {count}x: {len(string_list)} - {string_group}\n"

    # Remove the last comma and space from the result_string
    result_string = result_string + '\n '

    return result_string

def modelStrPathGen(clf, dirDict, n_splits, fit, randSeed):
    # Create a string to represent the model
    # Returns string for plotting and saving model to file.
    import re
    from sklearn.base import BaseEstimator

    # Create a string to represent the model in figure titles
    elements = []
    for name, step in clf.steps:
        elements.append(str(step))
    modelStr = ' -> '.join(elements)

    # Create a string to represent the model in filenames
    elements = []
    for name, step in clf.steps:
        if isinstance(step, BaseEstimator):
            elements.append(type(step).__name__)
        else:
            elements.append(str(step))
    figSaveStr = '_'.join(elements)

    # Create a string to represent the model save file
    elements = []
    for name, step in clf.steps:
        stepStr = f"{name}_{step}"
        elements.append(stepStr)

    modelParamStr = '_'.join(elements)
    ssDict = save_string_dict()
    for key, value in ssDict.items():
        modelParamStr = modelParamStr.replace(key, value)

    modelParamStr = modelParamStr + f"_CV{n_splits}"

    modelParamStr = re.sub(r'\n\s+', '', modelParamStr)
    modelStr = re.sub(r'\n\s+', '', modelStr)

    # use re.sub to remove the random state from the model string
    modelParamStr = re.sub(r',random_state=[^,]*,', f',randStateSeed={randSeed},', modelParamStr)
    modelStr = re.sub(r',random_state=[^,]*,', f',randStateSeed={randSeed},', modelStr)

    tmpDict = dict()
    tmpDict['tempDir_model'] = os.path.join(dirDict['tempDir_data'], modelParamStr)
    tmpDict['outDir_model'] = os.path.join(dirDict['outDir_data'], modelParamStr)

    for key, path in tmpDict.items():
        if not os.path.isdir(path):
            os.makedirs(path)

    dirDict.update(tmpDict)    
    dirDict['tempDir_outdata'] = os.path.join(tmpDict['tempDir_model'], f"{fit}_outdata.pkl")

    return modelStr, figSaveStr, dirDict

def dataStrPathGen(classifyDict, dirDict):
    # Create a string to represent the model
    # Returns string for plotting and saving model to file.

    ssDict = save_string_dict()
    conv_dict = create_drugClass_dict(classifyDict)

    # create a string for saving data
    keys_to_keep = ['data', 'label']

    if classifyDict['featureAgg']:
        keys_to_keep += ['featureAgg', 'featureSel_linkage', 'featureSel_distance', 'cluster_thres']

    if classifyDict['featurefilt']:
        keys_to_keep += ['featurefilt', 'filtType']

    # Add gridCV at the end
    if classifyDict['gridCV']:
        keys_to_keep.append('gridCV')

    smallDict = {key: value for key, value in classifyDict.items() if key in keys_to_keep}
    data_param_string = "-".join([f"{key}={value}" for key, value in smallDict.items()])

    # Abbreviate some of the features of the data string using a dictionary.
    for key, value in ssDict.items():
        data_param_string = data_param_string.replace(key, value)

    # Paths and directories - make directories for both figure outputs and temporary file locations
    tempDataStr = os.sep.join([dirDict['tempDir'], data_param_string])
    outDataStr = os.sep.join([dirDict['classifyDir'], data_param_string])
    dirDict['data_param_string'] = data_param_string

    tmpDir = dict()
    tmpDir['tempDir_data'] = tempDataStr
    tmpDir['outDir_data'] = outDataStr

    # Add a caching directory for the pipeline to speed up fitting
    classifyDict['tempDir_cacheDir'] = tmpDir['tempDir_data']

    # Cycle through the dict, and make sure each path exists
    for key, path in tmpDir.items():
        if not os.path.isdir(path):
            os.makedirs(path)

    dirDict.update(tmpDir)

    return classifyDict, dirDict

def save_string_dict():
    # Create a dictionary to allow for compressing of model and data names

    saveStringDict = dict()
    saveStringDict['label=class_'] = ''
    saveStringDict['label=drug'] = 'drug'
    saveStringDict['featurefilt=True-filtType=min'] = 'filtMin'
    saveStringDict['featureAgg=True'] = 'featAgg'
    saveStringDict['featureSel_linkage=average-featureSel_distance=correlation'] = 'avgCorrClus'
    saveStringDict['cluster_thres='] = 'clusThres'

    saveStringDict['n_jobs=-1'] = ''
    saveStringDict['n_workers=-1,  '] = ''
    saveStringDict['gridCV=True'] = 'gridCV'
    saveStringDict['featureTrans_PowerTransformer(standardize=False)'] = 'PowTrans'
    saveStringDict['featureSel'] = 'fSel'
    saveStringDict['featureScale_RobustScaler()'] = 'RobScal'
    saveStringDict['classif'] = 'clf'
    saveStringDict['BorutaFeatureSelector('] = 'BorFS('
    saveStringDict['MRMRFeatureSelector'] = 'MrmrFS'
    saveStringDict['n_features_to_select='] = ''
    saveStringDict['RobustScaler()'] = 'SelFroMod'
    saveStringDict['LogisticRegression('] = 'LogReg('
    saveStringDict["multi_class='multinomial'"] = 'multinom'
    saveStringDict[", solver='saga'"] = ''
    saveStringDict["\n\s+"] = ''

    return saveStringDict

def stringReportOut(selected_features_list, selected_features_params, YtickLabs, dirDict):
    from collections import Counter
    from plotFunctions import plot_histogram

    if len(YtickLabs) == 2:
        YtickLabs = [f"{YtickLabs[0]} vs {YtickLabs[1]}"]
    else:
        YtickLabs = [' vs '.join(YtickLabs)]

    f"Feature count per model {[len(x) for x in selected_features_list]}"

    # Report on which features make the cut.

    # Process the feature per model list into a string
    featurePerModel = [len(x) for x in selected_features_list]
    featurePerModelStr = str(featurePerModel)
    paramStr = ''

    if selected_features_params[0]:
        keyList = selected_features_params[0].keys()
        for key in list(keyList):
            keyVals = [x[key] for x in selected_features_params]
            paramStr += f"{key}: {str(keyVals)} \n"

    if np.sum(featurePerModel) == 0:
        return

    labels, counts = feature_model_count(selected_features_list)
    finalStr = conciseStringReport(labels, counts)

    # Plot the histogram of features per model
    # plot_histogram(featurePerModel, dirDict)

    print(f'==== {YtickLabs} ==== \n Features per Model: {featurePerModelStr}')
    print(f'Parameters: \n {paramStr}')
    print(f'Total Regions = {str(len(labels))} \n {finalStr}')

    file = open(os.path.join(dirDict['outDir_model'], 'featureSelReadout.txt'), 'w')
    file.write(f'==== {YtickLabs} ==== \n Features per Model: {featurePerModelStr} \n')
    file.write(f'Parameters: \n {paramStr}')
    file.write(f'Total Regions = {str(len(labels))} \n {finalStr}')
    file.close()

def feature_barplot(selected_features_list, selected_features_params, YtickLabs):
    from collections import Counter

    if len(YtickLabs) == 2:
        YtickLabs = [f"{YtickLabs[0]} vs {YtickLabs[1]}"]
    else:
        YtickLabs = [' vs '.join(YtickLabs)]


    # Report on which features make the cut.
    regionList = np.concatenate(selected_features_list[0])
    regionDict = dict(Counter(regionList))
    labels, counts = list(regionDict.keys()), list(regionDict.values())
    counts = np.array(counts)/len(selected_features_list[0])
    pd_df = pd.DataFrame({'Region': labels, 'Count': counts})
    pd_df = pd_df.sort_values(by='Count')
    
    pd_df.plot.barh(x='Region', y='Count', rot=0, figsize=(10, 20))
    plt.title(f'CV Split Feature Presence, Fraction: {YtickLabs}')
    plt.show()

def create_new_feature_names(dendrogram, original_feature_names):
    original_names = dendrogram['ivl']
    new_feature_names = []

    for i, name in enumerate(original_names):
        if name.isdigit():  # Check if the name is a cluster index
            cluster_index = int(name)
            children = dendrogram['dcoord'][cluster_index]
            child_names = [original_names[child] for child in children]
            combined_name = ', '.join(child_names)
            new_feature_names.append(combined_name)
        else:
            new_feature_names.append(original_feature_names[i])

    return new_feature_names

def simplified_name_trans_dict():
    simpDict = dict()
    simpDict['Psilocybin'] = 'PSI'
    simpDict['Ketamine'] = 'KET'
    simpDict['Entactogen'] = 'MDMA'
    simpDict['Acute SSRI'] = 'A-SSRI'
    simpDict['Chronic SSRI'] = 'C-SSRI'
    simpDict['6-Fluoro-DET'] = '6-F-DET'
    simpDict['Ag5HT2A'] = 'PSI + 5MEO'

    return simpDict

def feature_model_count(selected_features_list):
    # Returns the number of features in each model\
    from collections import Counter

    regionList = np.concatenate(selected_features_list)
    regionDict = dict(Counter(regionList))
    labels, counts = list(regionDict.keys()), list(regionDict.values())
    return labels, counts

def create_ABA_dict(dirDict):
        # Goes into the atlasDir looking for the specified files below, merges and filters them into a dictionary.
        
        # get Allen brain atlases
        ABA_tree = pd.read_csv(os.sep.join([dirDict['atlasDir'], 'ABA_CCF.csv']))
        ABA_tree = ABA_tree.rename(columns={'structure ID': 'id', '"Summary Structure" Level for Analyses': 'summary_struct'})
        ABA_tree = ABA_tree[['id', 'full structure name', 'abbreviation', 'Major Division', 'summary_struct']]

        ABA_hier = pd.read_csv(os.sep.join([dirDict['atlasDir'], 'ABAHier2017_csv.csv']))
        ABA_hier = ABA_hier.loc[:, ~ABA_hier.columns.str.contains('^Unnamed')]
        ABA_hier = ABA_hier.rename(columns={'Region ID': 'id', 'CustomRegion': 'Brain Area'})

        # merge the atlases
        ABA_tree = pd.merge(ABA_tree, ABA_hier, on=['id'])

        # define hierarchy by 'summary structure' from Wang/Ding . . . Ng, Cell 2020
        ABA_dict = ABA_tree[ABA_tree.summary_struct == 'Y']

        # remove ventricular systems, fiber tracts
        remove_list = ['Fibretracts', 'VentricularSystems']
        ABA_dict_filt = ABA_dict[~ABA_dict['Brain Area'].isin(remove_list)]

        # cleaning up and Merging with Data
        ABA_dict_filt = ABA_dict_filt.rename(columns={'id': 'Region ID'})
        ABA_dict_filt = ABA_dict_filt.drop(columns=['full structure name', 'Major Division', 'summary_struct'])

        return ABA_dict_filt

def retrieve_dict_data(dirDict, classifyDict):

    sys.path.append('../dependencies/')

    # Define the target directory
    targDir = dirDict['classifyDir']
    tagList = classifyDict['crossComp_tagList']

    # Report what is being looked for
    print(f"Looking for 'scoreDict.pkl' files in directories containing {tagList}")

    # Call the function and get the list of paths based on the tagList
    score_dict_paths = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(targDir):
        # Check if 'scoreDict.pkl' is present in the files
        if 'scoreDict_Real.pkl' in files:
            if all(tag in root for tag in tagList):
                score_dict_paths.append(os.path.join(root, 'scoreDict_Real.pkl'))

    assert len(score_dict_paths) > 0, f"No files found in {targDir} with the tag {tagList}"

    # Each directory name will be used to generate a label, based on the sequence between the strings in the directory name below
    startStr = tagList[0]
    endStr = '\PowerTrans'
    aucScores, meanScores, aucScrambleScores, meanScrambleScores = [], [], [], []
    featureLists, countNames  = [], []


    # Print the result
    print(f"Found 'scoreDict.pkl' files in directories containing {tagList}:")
    for path in score_dict_paths:

        # Load the scoreDict.pkl file and extract desired variables.
        with open(path, 'rb') as f:                 
            featureDict = pkl.load(f)
            featureLists.append(featureDict['featuresPerModel'])
            meanScores.append(np.mean(featureDict['scores']))
            aucScores.append(featureDict['auc']['Mean'])

        # Extract the label for the entry
        countNames.append(featureDict['compLabel'])

        # Load the scrambled dicts
        scramblePath = path.replace('_Real', '_Shuffle')
        with open(scramblePath, 'rb') as f:                 
            featureDict = pkl.load(f)

            meanScrambleScores.append(np.mean(featureDict['scores']))
            aucScrambleScores.append(featureDict['auc']['Mean'])

    return featureLists, countNames, aucScores, meanScores, aucScrambleScores, meanScrambleScores

def listToCounterFilt(listArray, filterByFreq=0):

    counter_u = Counter(listArray)
    
    if filterByFreq > 0:
        return Counter({k: v for k, v in counter_u.items() if v >= filterByFreq})
    else:
        return counter_u

def overlapCounter(list1, list2, filterByFreq=0):

    counter_u = listToCounterFilt(list1, filterByFreq)
    counter_v = listToCounterFilt(list2, filterByFreq)

    list1 = list(counter_u.keys())
    list2 = list(counter_v.keys())

    only_list1 = list(set(list1) - set(list2))
    only_list2 = list(set(list2) - set(list1))

    intersection = list(set(list1) & set(list2))
    
    return only_list1, only_list2, intersection

def sort_comparison_idx(orderedList, dataList):
    # Ordered list - a hardcoded list which has comparisons in the desired sequences - may include extras
    # dataList - a list representing the sequence of comparison data in data structures
    # function returns an index which resorts dataList and associated structures into the order in Ordered List

    orderedListNew = [name for name in orderedList if name in dataList]
    sort_indices = [dataList.index(name) for name in orderedListNew]

    return sort_indices

def weighted_jaccard_similarity(u, v, filt):

    counter_u, counter_v = Counter(u), Counter(v)

    # If Filt is non-0, filter out features in each counter whose count is not above it.
    if filt:
        counter_u = Counter({k: v for k, v in counter_u.items() if v > filt})
        counter_v = Counter({k: v for k, v in counter_v.items() if v > filt})

    intersection = sum((counter_u & counter_v).values())
    union = sum((counter_u | counter_v).values())

    # Using the modified Jaccard similarity with frequency
    similarity = intersection / union if union != 0 else 0

    return similarity

def feature_selection_info_gather(idx_o, clf, featureNames, penaltyStr, selected_features_list):
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
    selected_features_list[idx_o] = featureNamesSub

    return selected_features_list

def collect_shap_values(idx_o, explainers, shap_values_list, baseline_val, n_classes, clf, X_test_trans, feature_selected, test_index, featurePert):
    import shap
    
    # Select the correct explainer
    if len(clf._final_estimator.classes_) == 2:
        explainer = shap.LinearExplainer(clf._final_estimator, X_test_trans, feature_perturbation=featurePert)
    else:
        # Multiclass explainer must use interventional perturbation
        explainer = shap.LinearExplainer(clf._final_estimator, X_test_trans, feature_perturbation='interventional')

    explain_shap_vals = explainer.shap_values(X_test_trans)
    if len(clf._final_estimator.classes_) == 2:
        shap_values_test = [pd.DataFrame(explain_shap_vals, columns=feature_selected, index=test_index)]
    else:
        shap_values_test = [pd.DataFrame(x, columns=feature_selected, index=test_index) for x in explain_shap_vals]

    # Save output structures
    explainers[idx_o] = explainer
    for idx, shap_val in enumerate(shap_values_test):
        shap_values_list[idx].append(shap_val.reset_index())
        baseline_val[idx].append(explainer.expected_value)

    return explainers, shap_values_list, baseline_val

def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))  # Recursively flatten if it's a list
        else:
            flattened.append(item)
    return flattened

def generate_region_csv(lightSheetData, dirDict):

    # Save a list of the region_ids to a csv in atlasDir
    regionList = lightSheetData['Region_ID'].unique()

    # save the regionList to a csv
    regionList = pd.DataFrame(regionList, columns=['Region_ID'])
    regionList.to_csv(os.path.join(dirDict['atlasDir'], 'dataRegions.csv'), index=False)

    # Filter to brain region 'cortex'
    cortexData = lightSheetData[lightSheetData['Brain_Area'] == 'Cortex']
    regionList = pd.DataFrame(cortexData['Region_ID'].unique(), columns=['Region_ID'])
    # regionList.to_csv(os.path.join(dirDict['atlasDir'], 'dataRegions_Cortex.csv'), index=False)

def generate_ZscoreStructure(hierarchy_meso_ids, dirDict):
    #load data
    all_gene_data = pd.read_csv(os.sep.join([dirDict['atlasDir'], "allGeneStructureInfo_allgenes_summary_struct.csv"]))

    #getting structure averages
    all_gene_data_clean = all_gene_data.drop(columns=['Unnamed: 0', 'data_set_id', 'plane_of_section_id'])
    StructureAverages = all_gene_data_clean.groupby(['structure_id','gene_acronym']).mean().reset_index()
    # StructureAverageDensity = StructureAverage.groupby(['structure_id', 'gene_acronym']).mean().reset_index() # Two tables were identical, double check if done correctly FAQ
    # StructureAverages = pd.merge(StructureAverageEnergy, StructureAverageDensity)

    #averaging across whole brain
    all_gene_data_clean = all_gene_data_clean.drop(columns=['structure_id'])
    GeneAverageEnergy = all_gene_data_clean.groupby('gene_acronym').agg({'expression_energy': ['mean', 'std']}).reset_index()
    GeneAverageDensity = all_gene_data_clean.groupby('gene_acronym').agg({'expression_density': ['mean', 'std']}).reset_index()
    GeneAverages = pd.merge(GeneAverageEnergy, GeneAverageDensity)

    #calculate zscore for each brain structure
    #fresh copy of structure data
    ZscoreStructure = StructureAverages.copy()
    ZscoreStructure = ZscoreStructure.merge(GeneAverages, how='left', left_on='gene_acronym', right_on='gene_acronym')

    #rename columns for clarity
    column_indices = [4,5,6,7]
    new_names = ['brain_expression_energy_mean','brain_expression_energy_std','brain_expression_density_mean','brain_expression_density_std']
    old_names = ZscoreStructure.columns[column_indices]
    ZscoreStructure = ZscoreStructure.rename(columns=dict(zip(old_names, new_names)))

    #calculate zscore with structure/gene average and brain wide gene average/std
    ZscoreStructure['zscore'] = (ZscoreStructure['expression_energy'] - ZscoreStructure['brain_expression_energy_mean']) / (ZscoreStructure['brain_expression_energy_std'])

    #rename/clean for matching
    ZscoreStructure = ZscoreStructure.rename(columns={'structure_id':'Region_ID'})
    # ZscoreStructure = ZscoreStructure.drop(columns=['expression_energy','expression_density','brain_expression_energy_mean','brain_expression_energy_std','brain_expression_density_mean', 'brain_expression_density_std'])
    ZscoreStructure = ZscoreStructure.drop(columns=['expression_energy','expression_density','brain_expression_energy_mean','brain_expression_energy_std','brain_expression_density_mean', 'brain_expression_density_std'])

    #merge
    ZscoreStructure = ZscoreStructure.merge(hierarchy_meso_ids, on='Region_ID', how='inner', suffixes=('', '_x'))
    ZscoreStructure = ZscoreStructure.drop(ZscoreStructure.filter(regex='_x$').columns.tolist(),axis=1)

    return ZscoreStructure