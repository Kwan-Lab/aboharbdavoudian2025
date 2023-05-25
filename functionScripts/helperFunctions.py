import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Functions deployed elsewhere

def find_middle_occurrences(lst):
    from collections import defaultdict

    positions = defaultdict(list)
    first_occurrences = []
    middle_occurrences = []
    last_occurrences = []
    items = []

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

    positionDict = dict()
    for i in range(len(items)):
        positionDict[items[i]] = [first_occurrences[i], middle_occurrences[i], last_occurrences[i]]

    return positionDict

def agg_cluster(lightsheet_data, classifyDict, dirDict):
    from sklearn import datasets, cluster, preprocessing, linear_model
    from scipy.spatial import distance
    from scipy.cluster import hierarchy

    # Variables
    # dist_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    # linkage_list = ['ward', 'average', 'average', 'average', 'average']
    cluster_count = classifyDict['cluster_count']

    # Set variable for color coding output plots
    brainAreaList= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']
    brainAreaListPlot= ['Olfactory', 'Cortex', 'Hippo', 'Stri+Pall', 'Thalamus', 'Hypothalamus', 'Mid Hind Medulla', 'Cerebellum']
    brainAreaColor =     ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c'] #, '#dede00'
    brainAreaPlotDict = dict(zip(brainAreaList, brainAreaListPlot))
    brainAreaColorDict = dict(zip(brainAreaList, brainAreaColor))
    AreaIdx = dict(zip(brainAreaList, np.arange(len(brainAreaList))))

    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data.loc[:, colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea.loc[:, 'Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict[x] for x in regionArea.loc[:, 'Brain_Area']]
    regionArea.sort_values(by='abbreviation', inplace=True)

    clusterColors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c']
    colorDict = dict(zip(range(cluster_count), clusterColors))

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

    # Scale the data
    # if classifyDict['featureSel_scale']:
    #     RobScal = preprocessing.RobustScaler()
    #     X_scaled = RobScal.fit_transform(X)
    #     titleStr += ", Robust Scaled"

    #     sns.heatmap(X_scaled, square=True)
    #     plt.title(titleStr, fontsize=15)
    #     plt.show()

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

    # Save the figure in the dirDict['tempDir']
    plt.savefig(f"{dirDict['outDir']}{classifyDict['data']}_origDendrogram.png", bbox_inches='tight')

    plt.show()

    # Show the cluster map
    ax = sns.clustermap(X_scaled.T, row_linkage = col_row_link, col_cluster=False, square=True) #row_colors=[regionArea.Region_Color]
    plt.title(f"{titleStr}, Clustered", fontsize=15)
    plt.title(titleStr, fontsize=15)

    # Save the figure in the dirDict['tempDir']
    plt.savefig(f"{dirDict['outDir']}{classifyDict['data']}_origClustered.png", bbox_inches='tight')

    plt.show()

    # Generate new features based on clustering - average of each cluster
    # new_labels = hierarchy.fcluster(col_row_link, cluster_count, criterion='maxclust')  # Cluster based on how many clusters you want
    new_labels = hierarchy.fcluster(col_row_link, classifyDict['cluster_thres'], criterion='distance')
    unique_values, counts = np.unique(new_labels, return_counts=True)

    df_agged = pd.DataFrame(index=df_Tilted.index)

    # Process the features which are preserved first
    single_Idx = counts == 1
    unique_values[single_Idx]
    singleFeatureIdx = np.in1d(new_labels, unique_values[single_Idx])
    df_agged = df_agged.join(df_Tilted.loc[:, singleFeatureIdx])

    unique_values = unique_values[~single_Idx]
    counts = counts[~single_Idx]
    
    print(f'Clustering done: {len(counts)} clusters generated from {np.sum(counts)} features.')
    for idx, val in enumerate(unique_values):
        df_agged[f'Clus{idx+1}'] = round(df_Tilted[list(df_Tilted.columns[new_labels == val])].apply(lambda x: x.mean(), axis=1), 2)

    plt.figure(figsize=(30,10))
    sns.heatmap(df_agged.values, square=True)
    plt.title(f"{classifyDict['data']}, Clustered into {len(np.unique(new_labels))} Features", fontsize=15)
    plt.xlabel('Feature', fontsize=15)

    # Save the figure in the dirDict['tempDir']
    plt.savefig(f"{dirDict['outDir']}{classifyDict['data']}_clustered.png", bbox_inches='tight')

    plt.show()

    # Following the aggregation, check for multicollinearity
    # from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    # vif_val = pd.DataFrame({"Col":df_agged.columns})
    # vif_val["VIF"] = [vif(df_agged.values, i) for i in range(df_agged.shape[1])]
    # vif_val
    # print(vif_val)

    return df_agged

def create_region_to_area_dict(lightsheet_data, classifyDict):
    
    # For explicit sequence, set to false and modify.
    if 1:
        brainAreas = lightsheet_data.Brain_Area.unique()
    else:
        brainAreas= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']

    # Determines index and eventual sorting of brain areas. 
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))

    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data.loc[:, colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea.loc[:, 'Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)

    return regionArea

def reformatData(pandasdf, classifyDict):
    # Format the data for classification tasks
    # Performs filtering (Removing features based on percentile, clustering remaining features).

    # X of shape (n_samples, n_features), in our case (n_miceDrug, n_brain regions). y can be strings (they'll be drug names)
    import re

    # Each sample is a dataset, columns are abbreviations or full names, and the values can be counts or densities.
    pandasdf_Tilted = pandasdf.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])

    if not classifyDict['includeSAL']:
        pandasdf_Tilted = pandasdf_Tilted[~pandasdf_Tilted.index.str.contains('SAL')]

    if not classifyDict['include6FDET']:
        pandasdf_Tilted = pandasdf_Tilted[~pandasdf_Tilted.index.str.contains('6FDET')]

    # Convert y into different labels based on interest. 
    conv_dict = dict()

    match classifyDict['label']:
        case 'class_5HT2A':
            # 5-HT2A agonist psychedelics (Psilo, 5-MeO-DMT) vs entactogen (MDMA)
            conv_dict['PSI'] = 'Ag_5HT2A'
            conv_dict['DMT'] = 'Ag_5HT2A'
            conv_dict['MDMA'] = 'Entact'
        case 'class_5HTR':
            # Typtamines (Psilo, 5-MeO-DMT) vs non-Hallucinogenic trypamines (6-F-DET)
            conv_dict['PSI'] = 'H_Tryptamine'
            conv_dict['DMT'] = 'H_Tryptamine'
            conv_dict['6FDET'] = 'NH_Tryptamine'
        case 'class_Trypt':
            # 5-HT2A favored (Psilo) vs 5-HT1A favored (5-MeO-DMT)
            conv_dict['PSI'] = 'Ag_5HT2A'
            conv_dict['DMT'] = 'Ag_5HT1A'
        case 'class_Speed':
            # Fast vs Slow (Psi, 5-MeO-DMT, Ketamine vs Acute SSRI)
            conv_dict['PSI'] = 'Fast Acting'
            conv_dict['DMT'] = 'Fast Acting'
            conv_dict['KET'] = 'Fast Acting'
            conv_dict['A-SSRI'] = 'Slow Acting'
        case 'class_Psy_NMDA':
            # Fast Psychedelic vs Fast NMDA-R Agonist (Psi, 5-MeO-DMT vs Ketamine)
            conv_dict['PSI'] = 'Ag_5HT2A'
            conv_dict['DMT'] = 'Ag_5HT2A'
            conv_dict['KET'] = 'Ag_NMDA-R'
            
        case 'class_crash':
            # Drugs with 'low' afterwards vs those that dont (Ketamine, MDMA vs Psi, 5-MeO-DMT)
            print('d')
        case 'class_SSRI':
            # Acute vs Chronic SSRIs
            conv_dict['A-SSRI'] = 'Acute_SSRI'
            conv_dict['C-SSRI'] = 'Chronic_SSRI'

    # In all cases, translate the dataset index into a vector of drugs
    y = pd.Series([re.sub(r'\d+$', '', string) for string in pandasdf_Tilted.index])

    # Convert y based on dictionary above
    if classifyDict['label'] != 'drug':
        # Filter the table based on which labels you don't want
        boolean_array = np.array([string in conv_dict.keys() for string in y])
        pandasdf_Tilted = pandasdf_Tilted[boolean_array]

        # extract remaining entry labels and rename them.
        labelVec = pd.Series([re.sub(r'\d+$', '', string) for string in pandasdf_Tilted.index])
        pandasdf_Tilted.index = labelVec.map(conv_dict)

        # Create the new label vector
        y = pd.Series(pandasdf_Tilted.index)

        yNumDict = dict(zip(np.unique(y), np.arange(len(np.unique(y)))))

    else:
        # Convert y into numbers for use later. Convert it to a specific set
        orderedList = ['PSI', 'KET', 'DMT', '6FDET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']
        yNumDict = dict(zip(orderedList, np.arange(len(orderedList))))

    # develop an array of feature names
    featureNames = pandasdf_Tilted.columns

    # Convert y into numbers for use later. Convert it to a specific set
    numYDict = {value: key for key, value in yNumDict.items()}

    # Extract final values
    X = pandasdf_Tilted.reset_index(drop=True)
    X = np.array(X.values)
    
    y = np.array(y.map(yNumDict))

    return X, y, featureNames, numYDict

def  filter_features(pandasdf, classifyDict):
    # Takes in a pandas dataframe, returns a version where features are filtered and the remainder are aggregated.
    import re
    plotHist = 0
    # ======= Filter Phase ========
    # Filter based on percentile (get rid of very high or very low features)
    
    # Pivot data to represent samples, features, and data correctly for a heatmap.
    df_Tilted = pandasdf.pivot(index=classifyDict['feature'], columns='dataset', values=classifyDict['data'])

    # Visualize the flatten data
    if plotHist:
        plt.hist(df_Tilted.values.flatten(), bins=100, edgecolor='black')
        x_lim = plt.xlim()
        y_lim = plt.ylim()

        plt.xlim(0, x_lim[1])
        plt.ylim(y_lim[0], 20)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram Before')

    vmax = np.percentile(df_Tilted.values.flatten(), 99.9)
    remaining_features = np.array(df_Tilted.index[df_Tilted.max(axis=1) < vmax])
    pandasdf_filt = pandasdf[pandasdf[classifyDict['feature']].isin(remaining_features)]

    if plotHist:
        plt.hist(np.array(pandasdf_filt[classifyDict['data']]), bins=50, edgecolor='black')
        plt.show()
    
    print(f"feature count shifted from {len(df_Tilted.index)} to {len(remaining_features)}, removing all features with instance of '{classifyDict['data']}' over {round(vmax, 2)}")

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

