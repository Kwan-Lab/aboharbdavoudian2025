import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import exists, join
from math import isnan
from tqdm import tqdm

import matplotlib.patches as patches
import matplotlib.ticker as tkr
import helperFunctions as hf
import scipy.stats as stats
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import textwrap
import matplotlib.ticker as tkr
from statannotations.Annotator import Annotator
import configFunctions as config    

sys.path.append('dependencies')

def plot_headTwitchTotal(dirDict):

    htrDataPath = os.sep.join([dirDict['dataDir'],'behavioral','HTR_summary_data.csv'])
    df = pd.read_csv(htrDataPath)

    # Some parameters
    figHeight = 1.8
    figWidthHT = 4.15

    colorDict = hf.create_color_dict()
    linWidth = plt.rcParams['axes.linewidth']

    # Plot the box and whisker plot
    plt.figure(figsize=(figWidthHT, figHeight))

    plotOrder = ['Saline', '6-Fluoro-DET', 'Psilocybin', '5-MeO-DMT']
    tickLabels = ['SAL', '6-F-DET', 'PSI', '5MEO']

    colorPal = [colorDict[tickLabel] for tickLabel in tickLabels]

    boxprops = dict() # alpha=0.7
    ax = sns.boxplot(data=df, x= 'drug', y= 'total_HTR', order=plotOrder, palette=colorPal, boxprops=boxprops, linewidth=linWidth) 
    ax.legend().remove()
    sns.swarmplot(data=df, x= 'drug', y= 'total_HTR', color='black', size = 3, order=plotOrder, palette=colorPal)

    hf.extract_stats_per_box(df)

    # Stats - Place here to keep lines within the figure.
    pairs = [('Saline', 'Psilocybin'), ('6-Fluoro-DET', 'Psilocybin'), ('Saline', '5-MeO-DMT'), ('6-Fluoro-DET', '5-MeO-DMT')]
    annotator = Annotator(ax, pairs, data=df, x='drug', y='total_HTR', order=plotOrder)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    # Label the axes
    plt.ylim(0, 150)
    plt.xticks(ticks=[0, 1, 2, 3], labels=tickLabels)
    plt.yticks(np.arange(0, 126, 25))
    ax.yaxis.set_tick_params(which = 'both', length=1, width=linWidth)
    ax.xaxis.set_tick_params(which = 'both', length=1, width=linWidth)

    plt.ylabel('Head-twitch count', fontsize=7)
    plt.xlabel('') 

    savePath = os.path.join(dirDict['outDir'], 'HTR_total.svg')
    plt.savefig(savePath, format='svg', bbox_inches='tight')
    plt.show()

def plotTotalPerDrug(pandasdf, column2Plot, dirDict):
    # Select a random region to collect 'total_cells' from
    totalCellCountData = pandasdf[pandasdf.Region_ID == 88]

    totalCellCountData['sex'] = totalCellCountData['sex'].replace({'M': 'Male', 'F': 'Female'})

    # Shift the color codes to RGB and add alpha
    colorDict = hf.create_color_dict('drug', 0)

    scaleFactor = 1
    figSize = (3.2*scaleFactor, 1.278*scaleFactor)

    plt.figure(figsize=figSize)
    ax = sns.boxplot(x="drug", y=column2Plot, data=totalCellCountData, whis=0, dodge=False, showfliers=False, linewidth=.5, hue='drug', palette=colorDict)

    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))

    # Cylce through x-axis labels and change their color to match the boxplot
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_color(colorDict[label.get_text()])
    
    ax2 = sns.scatterplot(x="drug", y=column2Plot, data=totalCellCountData, hue='drug', linewidth=0, style='sex', markers=True, s=5, palette=colorDict, ax=ax, edgecolor='black')

    # remove legend
    # plt.legend([], [], frameon=False)
    # Delete all but the final 2 legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[-2:], labels=labels[-2:], loc='upper right')

    # cleanup
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.yaxis.set_tick_params(which = 'both', length=1.5, width=1)
    ax2.xaxis.set_tick_params(which = 'both', length=1, width=1)
    # ax2.yaxis.set_tick_params(which = 'both', length=1, width=0.5)
    # ax2.yaxis.set_tick_params(length=1, width=0.2)

    ax.set_yscale('log')
    ax.set(ylim=(7.9e5, 1e7))
    ax.set(xlabel='')
    ax.set_ylabel('c-Fos+ cell count', fontsize=7)
    # ax.set_ylabel()
    sns.despine()

    plt.savefig(dirDict['outDir'] + os.sep + 'totalCells', bbox_inches='tight')

    sns.set_theme()

def plotLowDimEmbed(pandasdf, column2Plot, dirDict, dimRedMeth, classifyDict, ldaDict):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler, PowerTransformer
    from itertools import combinations
    from sklearn.pipeline import Pipeline

    colorHex = hf.create_color_dict(dictType='drug')
    # Plot
    sns.set(font_scale=2)
    sns.set_style('ticks')
    sns.despine()
    
    # If some filtering is desired, do so here
    if classifyDict['featurefilt']:
        pandasdf = hf.filter_features(pandasdf, classifyDict)
        filtTag = 'filt'
    else:
        filtTag = ''

    # Pivot the lightsheet data table
    df_Tilted = pandasdf.pivot(index='dataset', columns='abbreviation', values=column2Plot)
    y = np.array([x[:-1] for x in df_Tilted.index])
    df_Tilted['y_vec'] = y

    n_comp = 2

    # Apply some preprocessing to mimic pipeline
    transMod = PowerTransformer(method='yeo-johnson', standardize=False)
    scaleMod = RobustScaler()

    if dimRedMeth == 'PCA':
        dimRedMod = PCA(n_components=n_comp)
        compName = 'Principal Component '
    elif dimRedMeth == 'LDA':
        dimRedMod = LDA(n_components=n_comp)
        compName = 'Linear discriminant '
    else:
        KeyError('dimRedMethod not recognized, pick LDA or PCA')

    pipelineList = [('transMod', transMod), ('scaleMod', scaleMod), ('dimRedMod', dimRedMod)]
    # pipelineList = [('scaleMod', scaleMod), ('dimRedMod', dimRedMod)]
    pipelineObj = Pipeline(pipelineList)

    ### Identify sets of training/testing data to loop through

    if not ldaDict:
        # Full set is used for training and testing
        analysisNames = 'All'
        trainSets = [set(df_Tilted.y_vec.unique())]
        testSets = [set(df_Tilted.y_vec.unique())]
    else:
        # Create a paired list of training sets and testing sets. 
        analysisNames = list(ldaDict.keys())
        trainSets = [ldaDict[aName][0] for aName in analysisNames]
        testSets = [ldaDict[aName][1] for aName in analysisNames]

    customOrder = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']

    config.setup_LDA_settings()

    for aName, trainSet, testSet in zip(analysisNames, trainSets, testSets):

        df_train = df_Tilted[df_Tilted.y_vec.isin(trainSet)]
        df_plot = df_Tilted[df_Tilted.y_vec.isin(testSet)]
        testOnly = [item for item in testSet if item not in trainSet]

        if len(trainSet) == 2 and dimRedMeth == 'LDA':
            pipelineList[2][1].n_components = 1
            colNames = [compName]
        else:
            pipelineList[2][1].n_components = n_comp
            colNames = [f"{compName}{x}" for x in range(1, n_comp+1)]

        # Fit on train data, then transform plot data
        pipelineObj.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
        df_plot_data_transformed = pipelineObj.transform(df_plot.iloc[:, :-1])

        # Create average cases for each drug.
        dimRedData = pd.DataFrame(data=df_plot_data_transformed, index=df_plot.index, columns=colNames)
        if df_plot_data_transformed.shape[1] == 1:
            # dimRedData['null'] = np.zeros(dimRedData.shape[0])
            dimRedData['null'] = np.random.rand(dimRedData.shape[0])

        dimRedData.loc[:, 'drug'] = pd.Categorical(y, categories=customOrder, ordered=True)
        dimRedDrugMean = dimRedData.groupby(by='drug').mean()

        # Means aren't sorted like the centers. Problems here are clear when the mean dot isn't the same color.
        if aName == 'All':
            resortIdx = [1, 2, 3, 0, 4, 5, 6, 7]
            dimRedDrugMean = dimRedDrugMean.iloc[resortIdx]

        if trainSet == testSet:
            pairs = list(combinations(range(n_comp), 2))
            for comp_pair in pairs:
                col1 = colNames[comp_pair[0]]
                col2 = colNames[comp_pair[1]]

                plt.figure(figsize=(2.25, 2.25))  # Adjust the figure size as needed
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData, s=10, alpha=0.75, palette=colorHex)
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean, s=20, legend=False, edgecolor='black', palette=colorHex)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)

                # Customize the plot
                # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
                # plt.title(f"Linear Discrimants of {aName}", fontsize=20)

                # Save
                plt.savefig(dirDict['outDir'] + os.sep + f"dimRed_{aName}_{col1} x {col2}", bbox_inches='tight')

                plt.show()
        else:
            pairs = list(combinations(range(n_comp), 2))
            for comp_pair in pairs:
                col1 = colNames[comp_pair[0]]
                col2 = colNames[comp_pair[1]]

                # Plot the training set, slightly lighter
                plt.figure(figsize=(2.25, 2.25))  # Adjust the figure size as needed
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData[dimRedData.drug.isin(trainSet)], s=10, alpha=0.5, legend=False, palette=colorHex)
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean[dimRedDrugMean.index.isin(trainSet)], s=20, alpha=0.5, legend=False, edgecolor='black', palette=colorHex)

                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData[dimRedData.drug.isin(testOnly)], s=12, edgecolor='black', palette=colorHex, marker="D")
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean[dimRedDrugMean.index.isin(testOnly)], s=25, legend=False, edgecolor='black', palette=colorHex, marker="D")
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)

                # Customize the plot
                # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
                # plt.title(f"Linear Discrimants of {aName}", fontsize=20)

                # Save
                plt.savefig(dirDict['outDir'] + os.sep + f"dimRed_{aName}_{col1} x {col2}", bbox_inches='tight')

                plt.show()            
        # else:
        #     plt.figure(figsize=(2.25, 2.25))  # Adjust the figure size as needed
        #     sns.scatterplot(x=compName, y='null', hue='drug', data=dimRedData, s=10, alpha=0.75, palette=colorHex)
        #     sns.scatterplot(x=compName, y='null', hue='drug', data=dimRedDrugMean, s=20, legend=False, edgecolor='black', palette=colorHex)
        #     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)

        #     # # Customize the plot
        #     # # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
        #     # # plt.title(f"Linear Discrimants of {aName}", fontsize=7)

        #     # Save
        #     plt.savefig(dirDict['outDir'] + os.sep + f"dimRed_{aName}_{filtTag}_{compName} x null", bbox_inches='tight')

        #     plt.show()
    # Reset changes made
    sns.set_theme()

def meanCountPerRegion(pandasdf):
    # Pull the mean count per region to help highlight.
    meanCountPerRegion = pandasdf.groupby(['Region_Name', 'Brain_Area'])['count', 'volume_(mm^3)'].mean().reset_index()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x='count', y='volume_(mm^3)', hue='Brain_Area', data=meanCountPerRegion)

    # Change the plot to have x and y axes limits at 80th percentile
    plt.xlim(0, np.percentile(meanCountPerRegion['count'], 95))
    plt.ylim(0, np.percentile(meanCountPerRegion['volume_(mm^3)'], 95))

    plt.show()

    print('d')

def histPrePostScale(pandasdf, dataPerPlot, dirDict):
    # Create a grid of histograms of columns in a pandas dataframe.
    from sklearn.preprocessing import RobustScaler, PowerTransformer

    outDirPath = os.path.join(dirDict['outDir'], 'featureScale')
    if not os.path.exists(outDirPath):
        os.mkdir(outDirPath)

    pivotTables = []
    for dataV in dataPerPlot:
        pivotTables.append(pandasdf.pivot(index='dataset', columns='abbreviation', values=dataV))

    scaledData = PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(pivotTables[1])
    scaledData_df = pd.DataFrame(scaledData, index=pivotTables[1].index, columns=pivotTables[1].columns)
    pivotTables.append(scaledData_df)
    
    scaledData = RobustScaler().fit_transform(pivotTables[2])
    scaledData_df = pd.DataFrame(scaledData, index=pivotTables[2].index, columns=pivotTables[2].columns)
    pivotTables.append(scaledData_df)

    featList = list(pivotTables[0].columns)
    featList = featList[0:1]

    for feat in featList:
        imgPath = os.path.join(outDirPath, f"scaleChain_{feat}.png")

        if os.path.exists(imgPath):
            continue

        fig, axes = plt.subplots(1, 4, figsize=(4*4, 2))

        axes[0].hist(pivotTables[0].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[0].title.set_text(f"{dataPerPlot[0]}: {feat}")
        axes[0].grid(False)

        axes[1].hist(pivotTables[1].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[1].title.set_text(f"{dataPerPlot[1]}: {feat}")
        axes[1].grid(False)

        axes[2].hist(pivotTables[2].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[2].title.set_text(f"yj norm: {feat}")
        axes[2].grid(False)

        axes[3].hist(pivotTables[3].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[3].title.set_text(f"Robust Scaled yj norm: {feat}")
        axes[3].grid(False)

        plt.savefig(os.sep.join([outDirPath, f"scaleChain_{feat}"]), bbox_inches='tight')

def distance_matrix(lightsheet_data, classifyDict, dirDict):
    import matplotlib.patheffects as PathEffects
    import numpy as np

    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import pairwise_distances
    
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c'] #, '#dede00'

    dist_met_list = ['euclidean', 'minkowski', 'cityblock', 'seuclidean', 'mahalanobis', 'cosine']
    df_Tilted = lightsheet_data.pivot(index=classifyDict['feature'], columns='dataset', values=classifyDict['data'])

    for dist_met in dist_met_list:

        pairwise = pd.DataFrame(squareform(pdist(df_Tilted, metric=dist_met)),columns = df_Tilted.index,index = df_Tilted.index)

        plt.figure(figsize=(10,10))
        sns.heatmap(
            pairwise,
            # cmap='OrRd',
            # linewidth=1
        )
        plt.title(dist_met, fontsize=45)

        plt.savefig(dirDict['outDir'] + os.sep + f'Dist_{dist_met}_raw.png', format='png', bbox_inches='tight')

        sns.clustermap(pairwise, 
                       cmap='rocket', 
                       fmt='.2f', 
                        dendrogram_ratio = 0.1)

        plt.title(dist_met, fontsize=45)
        plt.savefig(dirDict['outDir'] + os.sep + f'Dist_{dist_met}_clustered.png', format='png', bbox_inches='tight')

        plt.show()

def correlation_plot(lightsheet_data, classifyDict, dirDict):
    # Create an index for sorting the brain Areas. List below is for custom ordering.
    # brainAreaList = lightsheet_data['Brain_Area'].unique().tolist()
    brainAreaColorDict = hf.create_color_dict(dictType='brainArea', rgbSwitch=0)
    brainAreaPlotDict = hf.create_brainArea_dict('short')
    regionArea = hf.create_region_to_area_dict(lightsheet_data, classifyDict['feature'])
    regionArea['Region_Color'] = regionArea['Brain_Area'].map(brainAreaColorDict)

    df_Tilted = lightsheet_data.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])
    df_Tilted = df_Tilted.reindex(regionArea[classifyDict['feature']].tolist(), axis=1)

    # Calculate spearman and pearson correlation.
    corr_matrix = np.corrcoef(df_Tilted.values.T)
    s_corr_matrix = stats.spearmanr(df_Tilted.values)

    corr_mats = [corr_matrix, s_corr_matrix[0]]
    corr_names = ['Pearson', 'Spearman']

    # Plotting variables
    scalefactor = 12
    cmap = 'rocket'
    yticklabels = df_Tilted.columns.tolist()
    plotDim = len(yticklabels) * scalefactor * 0.015

    for corr_data, corr_name in zip(corr_mats, corr_names):
        # Plotting
        plt.figure(figsize=(plotDim*1.1, plotDim))
        # cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        ax = sns.heatmap(corr_data, cmap=cmap, fmt='.2f', yticklabels=yticklabels, xticklabels=yticklabels, square=True)

        # Color code the labs
        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            colorCode = regionArea[regionArea[classifyDict['feature']] == ytick._text]['Region_Color'].values[0]
            xtick.set_color(colorCode)
            ytick.set_color(colorCode)

        # Adjust the color bar
        cbar = ax.collections[0].colorbar  # Get the colorbar object
        cbar.ax.tick_params(labelsize=45, width=1)  # Increase font size and adjust width

        # Add in vertical lines breaking up sample types
        # _, line_break_ind = np.unique(regionArea.Brain_Area, return_index=True)
        # for idx in line_break_ind:
        #     plt.axvline(x=idx, color='black', linewidth=6)
        #     plt.axhline(y=idx, color='black', linewidth=6)

        # Add additional text to denote which area each region belongs to
        positionDict = hf.find_middle_occurrences(regionArea.Brain_Area)
        mid_idx = [x[1] for x in positionDict.values()]
        items = positionDict.keys()
        for mid_sample_ind, label in zip(mid_idx, items):
            ax.text(mid_sample_ind, corr_matrix.shape[0] + 5, f"{brainAreaPlotDict[label]}", size = 30, ha='center', va='top', color=brainAreaColorDict[label]) #, transform=ax.transAxes
        
        # Add colored boxes
        for area in items:
            squarePos = positionDict[area][0]
            squareLen = positionDict[area][-1] - positionDict[area][0] + 1
            tmpVar = patches.Rectangle((squarePos, squarePos), squareLen, squareLen, linewidth=7, edgecolor='black', fill=False)
            # tmpVar = patches.Rectangle((squarePos, squarePos), squareLen, squareLen, linewidth=7, edgecolor=brainAreaColorDict[area], fill=False)
            ax.add_patch(tmpVar)

        # Labels    
        titleStr = f"{corr_name}, {classifyDict['data']} Correlations Across Regions"

        ax.set_ylabel("Region", fontsize=40)
        ax.set_xlabel('Region', fontsize=45, labelpad=40)
        plt.tick_params(axis='x', which='both', length=0)
        plt.title(titleStr, fontsize=65)
        
        plt.savefig(dirDict['classifyDir'] + os.sep + titleStr + '.png', format='png', bbox_inches='tight')
        plt.show()

def correlation_plot_hier(lightsheet_data, classifyDict, dirDict):
    # Plot Hierarchical Correlation Clustering Heatmap

    brainAreaColorDict = hf.create_color_dict('brainArea', 0)
    brainAreas = list(brainAreaColorDict.keys())
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))

    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data[colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea['Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict[x] for x in regionArea['Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)

    df_Tilted = lightsheet_data.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])
    df_Tilted = df_Tilted.reindex(regionArea[classifyDict['feature']].tolist(), axis=1)

    corrHier = True
    if not corrHier:
        ylabelStr = "Dataset"

        # Cluster the data directly
        ax = sns.clustermap(df_Tilted.values, cmap='rocket', fmt='.2f', yticklabels=df_Tilted.index, xticklabels=df_Tilted.columns, dendrogram_ratio = 0.1, figsize=(40, 10))
        ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xmajorticklabels(), fontsize = 3)
        ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_ymajorticklabels(), fontsize = 8)

        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            colorCode = regionArea[regionArea[classifyDict['feature']] == ytick._text]['Region_Color'].values[0]
            xtick.set_color(colorCode)
            ytick.set_color(colorCode)

        # Adjust the color bar
        
        ax.ax_cbar.tick_params(width=1)  # Increase font size and adjust width

    else:
        ylabelStr = "Region"
        corr_matrix = np.corrcoef(df_Tilted.values.T)
        ax = sns.clustermap(corr_matrix, cmap='rocket', fmt='.2f', yticklabels=df_Tilted.columns, xticklabels=df_Tilted.columns, dendrogram_ratio = 0.1, figsize=(40, 40))

    # Labels    
    titleStr = f"{classifyDict['data']} - Clustering Across Regions"
    plt.ylabel(ylabelStr)
    plt.xlabel('Region')
    plt.tick_params(axis='x', which='both', length=0)
    plt.title(titleStr)
    
    plt.savefig(dirDict['classifyDir'] + os.sep + titleStr + '.png', format='png', bbox_inches='tight')
    plt.show()

def correlation_subset(processed_data, lightsheet_data, modelCountDict, threshold, classifyDict, dirDict):
    # Generates a correlation matrix heatmap based on the correlation between specific features within a comparison.

    brainAreaColorDict = hf.create_color_dict('brainArea', 0)
    brainAreas = list(brainAreaColorDict.keys())
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))

    # Use the unprocessesd lightsheet_data to generate a structure for sorting the processed data
    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data[colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea['Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict[x] for x in regionArea['Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)

    # Filter the lightsheet_data to only include data from the drug column which matches condition
    # df = lightsheet_data[lightsheet_data['drug'] == drug]
    df = processed_data

    # identify which regions in modelCount have higher or equal to the threshold
    regionKeepIdx = np.array(np.array(modelCountDict[1]).astype(int) > threshold)
    regionList = np.array(modelCountDict[0])[regionKeepIdx]
    df_Tilted = df[regionList]

    corr_matrix = np.corrcoef(df_Tilted.values.T)
    # ax = sns.heatmap(corr_matrix, cmap='rocket', fmt='.2f', yticklabels=regionList, xticklabels=regionList, square=True)
    ax = sns.clustermap(corr_matrix, cmap="vlag", center=0, fmt='.2f', yticklabels=df_Tilted.columns, xticklabels=df_Tilted.columns, dendrogram_ratio = 0.1, figsize=(10, 10))

    regionArea.set_index('abbreviation', inplace=True)
    
    for tick_label in ax.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        regionColor = regionArea.loc[tick_text, 'Region_Color']
        tick_label.set_color(regionColor)

    for tick_label in ax.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        regionColor = regionArea.loc[tick_text, 'Region_Color']
        tick_label.set_color(regionColor)
        
    # Label and save the image
    titleStr = f"Region {classifyDict['data']} correlation between {classifyDict['label']}"
    plt.tick_params(axis='x', which='both', length=0)
    plt.title(titleStr, fontsize=10)
    
    plt.savefig(os.sep.join([dirDict['outDir_model'], titleStr + '.svg']), format='svg', bbox_inches='tight')
    plt.show()

def plot_data_heatmap(lightsheet_data, heatmapDict, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # TODO: shift code to use 'GridSpec' and create a single image with 3 equally sized columns and a colorbar at once.
    # Creates the heatmap for the data
    # heatmapDict['feature'] = 'abbreviation'
    # heatmapDict['data'] = 'cell_density', 'count', 'count_norm', 'density_norm', 'count_norm_scaled'

    # Set variables
    colorMapCap = True
    dataFeature = heatmapDict['feature']
    dataValues = heatmapDict['data']
    blockCount = heatmapDict['blockCount']

    # take the mean across all the datasets for each region, then log transform
    if heatmapDict['logChangeSal'] == True:
        # Create a version of the dataset which is averaged and adjusted against SAL. 
        df_avg  = lightsheet_data.groupby([dataFeature, 'drug'])[dataValues].mean().reset_index()
        df_piv = df_avg.pivot(index=dataFeature, columns='drug', values=dataValues)
        df_piv = df_piv.div(df_piv['SAL'], axis=0)
        # df_piv.drop(columns=['SAL'], inplace=True)
        df_plot = np.log(df_piv)
        
    else:
        # Pivot data to represent samples, features, and data correctly for a heatmap.
        df_plot = lightsheet_data.pivot(index=dataFeature, columns='dataset', values=dataValues)

    # Resort for coherence across figures
    reIdx = heatmapDict['SortList']
    if heatmapDict['logChangeSal'] == False:
        # Identify the highest dataset number and use that to sort things. 
        # Will throw an error if dataset numbers are not the same across conditions.
        maxSampleNum = np.max(np.array([x[-1:] for x in list(lightsheet_data.dataset)]).astype(int)) + 1
        reIdx = [f'{item}{i}' for item in reIdx for i in range(1, maxSampleNum)]
    df_plot = df_plot[reIdx]

    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, dataFeature)

    # Create indicies for dividing the data into the correct number of  sections regardless of the size
    row_idx_set = np.zeros((blockCount, 2), dtype=int)
    if heatmapDict['areaBlocks'] == True:
        # Make the blocks 2 roughly equal size blocks
        line_break_num, line_break_ind = np.unique(regionArea.Brain_Area_Idx, return_index=True)
        row_idx_set[0,:] = [line_break_ind[0], line_break_ind[5]]
        row_idx_set[1,:] = [line_break_ind[5], len(df_plot)]
    
    else:
        indices = np.linspace(0, len(df_plot), num=blockCount+1, dtype=int)
        for block_idx in range(blockCount):
            row_idx_set[block_idx][0] = indices[block_idx]
            row_idx_set[block_idx][1] = indices[block_idx+1]
    

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]

    # Find the ends of the colormap
    if colorMapCap:
        vmin, vmax = np.percentile(df_plot.values.flatten(), [1, 99])
    else:
        vmin = df_plot.min().min()
        vmax = df_plot.max().max()

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # Create a version with and without the colorbar for purposes of keeping segments equally sized.
    colorBarSwitch = [True, False]

    for cbs in colorBarSwitch:

        if heatmapDict['logChangeSal'] == True:
            cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
            scalefactor = 5
            figH = (scalefactor*8)/blockCount
            figW = blockCount * 3
        else:
            cmap = 'rocket'
            scalefactor = 12
            figH = (scalefactor*5)/blockCount
            figW = blockCount * 10

        fig, axes = plt.subplots(1, blockCount, figsize=(figW, figH))  # Adjust figsize as needed
        # figsize=(scalefactor*2.4, len(df_plot)/len(row_idx_set) * scalefactor * 0.0125)

        if blockCount == 1:
            axes = [axes]

        for idx, row_set in enumerate(row_idx_set):

            # Slice and modify previous structures to create segment
            df_plot_seg = df_plot.iloc[row_set[0]: row_set[1], :]
            regionArea_local = regionArea[regionArea[dataFeature].isin(df_plot_seg.index)]
            region_idx = regionArea_local.Brain_Area_Idx  # Extract for horizontal lines in plot later.

            matrix = df_plot_seg.values

            xticklabels = df_plot_seg.columns.values.tolist()
            yticklabels = df_plot_seg.index.values.tolist()
            if heatmapDict['logChangeSal'] == True:

                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx] , fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=xticklabels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter}, center=0)
                horzLineColor = 'black'

            else:
                xticklabels = [x[0:-1] for x in xticklabels]
                # Convert to x axis labels
                x_labels = ['' for _ in range(matrix.shape[1])]
                result = hf.find_middle_occurrences(xticklabels)
                for mid_sample_ind in result:
                    x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
                
                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx], fmt='.2f', cbar = cbs, square=True, yticklabels=yticklabels, xticklabels=x_labels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter})
    
                # Clear the x-ticks
                heatmap.tick_params(axis='x', which='both', length=0, labelsize=14)

                # Add in vertical lines breaking up sample types
                _, line_break_ind = np.unique(xticklabels, return_index=True)
                for l_idx in line_break_ind:
                    axes[idx].axvline(x=l_idx, color='white', linewidth=1)
                horzLineColor = 'white'

            # Change
            # Shift the color codes to RGB and add alpha
            colorDict = hf.create_color_dict('drug', 0)
            # Cylce through x-axis labels and change their color to match the boxplot
            for _, label in enumerate(heatmap.get_xticklabels()):
                if label.get_text():
                    label.set_color(colorDict[label.get_text()])

            # Add in horizontal lines breaking up brain regions types.
            line_break_num, line_break_ind = np.unique(region_idx, return_index=True)
            for l_idx in line_break_ind[1:]:
                axes[idx].axhline(y=l_idx, color=horzLineColor, linewidth=1)
                
            # Set the ylabel on the first subplot.
            if idx == 0:
                axes[idx].set_ylabel("Region Names", fontsize=20)

            # if idx == 2:
            #     cbar = heatmap.collections[0].colorbar
            #     cbar.set_label('Colorbar Label', rotation=270, labelpad=5)

        titleStr = f"Data_{dataValues}_block_colorbar_{cbs}"  
        fig.suptitle(titleStr, fontsize=30, y=1.02)
        # fig.text(0.5, -.02, "Samples Per Group", ha='center', fontsize=20)
        plt.tight_layout(h_pad = 0, w_pad = .5)

        # Change the axis of the colorbar to represent multiples of 
        plt.savefig(dirDict['outDir'] + os.sep + f"{titleStr}", bbox_inches='tight')
        plt.show()

def plot_data_heatmap_perArea(lightsheet_data, heatmapDict, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # TODO: shift code to use 'GridSpec' and create a single image with 3 equally sized columns and a colorbar at once.
    # Creates the heatmap for the data
    # heatmapDict['feature'] = 'abbreviation'
    # heatmapDict['data'] = 'cell_density', 'count', 'count_norm', 'density_norm', 'count_norm_scaled'

    # Set variables
    colorMapCap = True
    dataFeature = heatmapDict['feature']
    dataValues = heatmapDict['data']
    blockCount =heatmapDict['blockCount']

    # take the mean across all the datasets for each region, then log transform
    if heatmapDict['logChangeSal'] == True:
        # Create a version of the dataset which is averaged and adjusted against SAL. 
        df_avg  = lightsheet_data.groupby([dataFeature, 'drug'])[dataValues].mean().reset_index()
        df_piv = df_avg.pivot(index=dataFeature, columns='drug', values=dataValues)
        df_piv = df_piv.div(df_piv['SAL'], axis=0)
        # df_piv.drop(columns=['SAL'], inplace=True)
        df_plot = np.log(df_piv)
        
    else:
        # Pivot data to represent samples, features, and data correctly for a heatmap.
        df_plot = lightsheet_data.pivot(index=dataFeature, columns='dataset', values=dataValues)

    # Resort for coherence across figures
    reIdx = heatmapDict['SortList']
    if heatmapDict['logChangeSal'] == False:
        # Identify the highest dataset number and use that to sort things. 
        # Will throw an error if dataset numbers are not the same across conditions.
        maxSampleNum = np.max(np.array([x[-1:] for x in list(lightsheet_data.dataset)]).astype(int)) + 1
        reIdx = [f'{item}{i}' for item in reIdx for i in range(1, maxSampleNum)]
    df_plot = df_plot[reIdx]

    # Create indicies for dividing the data into the correct number of  sections regardless of the size
    row_idx_set = np.zeros((blockCount, 2), dtype=int)
    indices = np.linspace(0, len(df_plot), num=blockCount+1, dtype=int)
    for block_idx in range(blockCount):
        row_idx_set[block_idx][0] = indices[block_idx]
        row_idx_set[block_idx][1] = indices[block_idx+1]
        
    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, dataFeature)

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]

    # Find the ends of the colormap
    if colorMapCap:
        vmin, vmax = np.percentile(df_plot.values.flatten(), [1, 99])
    else:
        vmin = df_plot.min().min()
        vmax = df_plot.max().max()

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # Create a version with and without the colorbar for purposes of keeping segments equally sized.
    colorBarSwitch = [True, False]

    for cbs in colorBarSwitch:

        if heatmapDict['logChangeSal'] == True:
            cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
            scalefactor = 12
            figH = (scalefactor*5)/blockCount
            figW = blockCount * 3
        else:
            cmap = 'rocket'
            scalefactor = 12
            figH = (scalefactor*5)/blockCount
            figW = blockCount * 10

        fig, axes = plt.subplots(1, blockCount, figsize=(figW, figH))  # Adjust figsize as needed
        # figsize=(scalefactor*2.4, len(df_plot)/len(row_idx_set) * scalefactor * 0.0125)

        if blockCount == 1:
            axes = [axes]

        for idx, row_set in enumerate(row_idx_set):

            # Slice and modify previous structures to create segment
            df_plot_seg = df_plot.iloc[row_set[0]: row_set[1], :]
            regionArea_local = regionArea[regionArea[dataFeature].isin(df_plot_seg.index)]
            region_idx = regionArea_local.Brain_Area_Idx  # Extract for horizontal lines in plot later.

            matrix = df_plot_seg.values

            xticklabels = df_plot_seg.columns.values.tolist()
            yticklabels = df_plot_seg.index.values.tolist()
            if heatmapDict['logChangeSal'] == True:

                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx] , fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=xticklabels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter}, center=0)
                horzLineColor = 'black'

            else:
                xticklabels = [x[0:-1] for x in xticklabels]
                # Convert to x axis labels
                x_labels = ['' for _ in range(matrix.shape[1])]
                result = hf.find_middle_occurrences(xticklabels)
                for mid_sample_ind in result:
                    x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
                
                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx], fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=x_labels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter})
    
                # Clear the x-ticks
                heatmap.tick_params(axis='x', which='both', length=0, labelsize=14)

                # Add in vertical lines breaking up sample types
                _, line_break_ind = np.unique(xticklabels, return_index=True)
                for l_idx in line_break_ind:
                    axes[idx].axvline(x=l_idx, color='white', linewidth=1)
                horzLineColor = 'white'

            # Add in horizontal lines breaking up brain regions types.
            line_break_num, line_break_ind = np.unique(region_idx, return_index=True)
            for l_idx in line_break_ind[1:]:
                axes[idx].axhline(y=l_idx, color=horzLineColor, linewidth=1)
                
            # Set the ylabel on the first subplot.
            if idx == 0:
                axes[idx].set_ylabel("Region Names", fontsize=20)

            # if idx == 2:
            #     cbar = heatmap.collections[0].colorbar
            #     cbar.set_label('Colorbar Label', rotation=270, labelpad=5)

        titleStr = f"Data_{dataValues}_block_colorbar_{cbs}"  
        fig.suptitle(titleStr, fontsize=30, y=1.02)
        fig.text(0.5, -.02, "Samples Per Group", ha='center', fontsize=20)
        plt.tight_layout(h_pad = 0, w_pad = .5)

        # Change the axis of the colorbar to represent multiples of 

        plt.savefig(dirDict['classifyDir'] + os.sep + f"{titleStr}", bbox_inches='tight')
        plt.show()

def create_heatmaps_allC(matrix, dim_to_loop=0, titleStatic='Heatmap', titleLoop=[], dirDict=[]):
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.colors import ListedColormap

    cmap = plt.cm.Blues

    loopRange = matrix.shape[dim_to_loop]

    if titleLoop == []:
        titleLoop = range(loopRange)

    fig, axes = plt.subplots(nrows=1, ncols=loopRange,
                             figsize=(loopRange * 1.5, 15))

    fullTitleStr = f"{titleStatic} Classification via L1 Regularization, Feature weights"

    fig.suptitle(fullTitleStr, fontsize=16, y=0.92)

    for i in range(loopRange):
        heatmap_data = matrix[i, :, :]
        plt.figure(figsize=(2, 10))

        # Only colorbar for the value all the way to the right
        sns.heatmap(heatmap_data, cmap=cmap,
                    cbar=False, fmt='.2f', ax=axes[i])

        axes[i].set_title(f"C = {str(titleLoop[i])}")

        axes[i].set_xlabel("Split")

        if i == 0:
            axes[i].set_ylabel("Feature")
        else:
            axes[i].tick_params(left=False, labelleft=False)

    plt.savefig(dirDict['classifyDir'] + os.sep + fullTitleStr, bbox_inches='tight')

    # Add a colorbar on the far right plot
    vmin = np.min(matrix)
    vmax = np.max(matrix)

    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])

    # [left, bottom, width, height] of the colorbar axis
    cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.show()

def create_heatmaps_perDrug(matrix, titleStatic='Heatmap', titleLoop=[], xLab = [], perPlotXTicks=[], dirDict=[]):
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # Create a TwoSlopeNorm normalization centered at 0
    vmin, vmax = matrix.min(), matrix.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Pull the color map, and set 0 to white
    cmap = sns.color_palette("rocket", as_cmap=True)  # Use the viridis color map as a base
    linSpaceInts = np.linspace(vmin, vmax, cmap.N*2)
    zero_idx = np.argmin(np.abs(linSpaceInts))
    colors = cmap(linSpaceInts)  # Extract the colors from the color map

    # Attempting to set the 0 value and the numbers around to to grey does not make the final result grey. 
    # zero_color = [.75, 0.75, 0.75, 1]
    # colors[zero_idx - 1] = zero_color
    # colors[zero_idx] = zero_color
    # colors[zero_idx + 1] = zero_color

    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    loopRange = matrix.shape[0]
    
    if titleLoop == []:
        titleLoop = range(loopRange)

    fig, axes = plt.subplots(nrows=1, ncols=loopRange,figsize=(loopRange * 1.5, 18))

    fullTitleStr = f"{titleStatic} Classification via L1 Regularization, Median Feature weights"

    fig.suptitle(fullTitleStr, fontsize=16, y=0.92)

    for i in range(loopRange):
        heatmap_data = matrix[i, :, :]
        plt.figure(figsize=(2, 10))

        # Only colorbar for the value all the way to the right
        sns.heatmap(heatmap_data, cmap=custom_cmap, vmin=vmin, vmax=vmax, cbar=False, fmt='.2f', ax=axes[i], norm=norm)

        axes[i].set_title(f"{str(titleLoop[i])}")

        axes[i].set_xlabel(xLab)
        perPlotXTicks = [str(x) for x in perPlotXTicks]
        # perPlotXTicks = [int(num) if isinstance(num, int) else str(num) if isinstance(num, float) else None for num in perPlotXTicks]
        axes[i].set_xticks(np.arange(0, len(perPlotXTicks))+0.5)
        axes[i].set_xticklabels(perPlotXTicks, fontdict={'fontsize':7})

        if i == 0:
            axes[i].set_ylabel("Feature")
            axes[i].set_yticklabels(labels=axes[i].get_yticklabels(), fontdict={'fontsize':8})
        else:
            axes[i].tick_params(left=False, labelleft=False)

        for idx in range(1, len(perPlotXTicks)): 
            axes[i].axvline(idx, color='black', linewidth=0.5)


    # Add a colorbar on the far right plot
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm) #
    sm.set_array([])

    # Not working
    # plt.savefig(dirDict['classifyDir'] + os.sep + fullTitleStr + '.png',
    #             format='png', bbox_inches='tight')

    # [left, bottom, width, height] of the colorbar axis
    cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig(dirDict['classifyDir'] + os.sep + fullTitleStr + '.png', format='png', bbox_inches='tight')
    plt.show()

def plot_cFos_delta(cfos_diff, drugList, fileOutName):
    plt.figure(figsize=(20,50))
    drugPairList = [(a + '-'+ b) for idx, a in enumerate(drugList) for b in drugList[idx + 1:]]
    palette = sns.color_palette(n_colors=len(drugList))

    for drug_comp_i in range(0, len(cfos_diff)):
        #melt data for plotting and specify color
        data = pd.melt(cfos_diff[drug_comp_i], id_vars=['Region_Name'], value_vars=drugPairList[drug_comp_i],
                        var_name='Drug', value_name='Change')
        if drug_comp_i == 0:
            data_melted = data
        else:
            data_melted = pd.concat([data_melted, data], ignore_index=True, sort=False)

    # Focus on the comparisons involving SAL condition
    drugDataMelt = data_melted[data_melted.Drug.str.contains('SAL')]

    #point plot
    ax = sns.pointplot(y='Region_Name', x='Change', data = drugDataMelt, errorbar=('ci', 95), join=False, units=16, errwidth = 0.5,
                    hue='Drug', palette = palette, dodge=0.4, scale=0.5)

    #cleanup
    sns.despine()
    plt.xlabel('cFos density change (%)')
    plt.ylabel('')

    plt.axvline(x=0, color='grey', linestyle='--', lw=0.5)

    fig = plt.savefig(fileOutName, bbox_inches='tight')

def plot_cFos_delta_new(lightsheet_data, cfos_diff, cfos_diff_labels, drugList, fileOutName):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    data_melted = pd.DataFrame()
    for data, label in zip(cfos_diff, cfos_diff_labels):
        # Melt data for plotting and specify color
        dataMelt = pd.melt(data, id_vars=['Region_Name'], value_vars=label, var_name='Drug', value_name='Change')
        data_melted = pd.concat([data_melted, dataMelt], ignore_index=True, sort=False)

    # Focus on the comparisons involving SAL condition
    drugDataMelt = data_melted[data_melted.Drug.str.contains('SAL')]
    drugDataMelt['Drug'] = drugDataMelt['Drug'].str.replace('-SAL', '')

    # Testing - leave only KET and PSI
    drugDataMelt = drugDataMelt[drugDataMelt['Drug'].isin(['KET', 'PSI', '5MEO'])]

    # Generate the dictionary for region_name to brain_area
    brainAreaColorDict = hf.create_region_to_area_dict(lightsheet_data, ['Region_Name', 'Region_ID'])

    # merge the brainAreaColorDict with the drugDataMelt
    drugDataMelt = drugDataMelt.merge(brainAreaColorDict, left_on='Region_Name', right_on='Region_Name')

    # Switch drugs to categorical variables
    drugDataMelt['Drug'] = pd.Categorical(drugDataMelt['Drug'], categories=['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL'], ordered=True)
    palette = hf.create_color_dict(dictType='drug')
                      
    # Cycle through brainAreas, filter drugDataMelt, and plot
    # for brainArea in brainAreas:

    # brainAreadata = drugDataMelt[drugDataMelt['Brain_Area'] == brainArea].sort_index()
    brainAreadata = drugDataMelt.sort_index()
    regionCount = brainAreadata.Region_Name.unique().shape[0]
    
    # Identify the index for the supraoptic nucleus
    supraoptic_idx = brainAreadata[brainAreadata['Region_Name'] == 'Supraoptic nucleus'].index[0]
    plot_idx_set = [[0, supraoptic_idx], [supraoptic_idx, len(brainAreadata)]]
    
    # Create a figure with two columns
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, regionCount*0.05), sharey=True)

    for plot_idx, data_idx in enumerate(plot_idx_set):

        plt.figure(figsize=(1.5, regionCount*0.03))

        # Plot in the first column
        ax = sns.pointplot(y='Region_Name', x='Change', data=brainAreadata.iloc[data_idx[0]:data_idx[1]], errorbar=('ci', 95), legend=False,
                      join=False, units=16, errwidth=0.5, hue='Drug', palette=palette, dodge=0.4, scale=0.25)

        # Limit the x axis to 500
        lowLim = -100
        upperLim = min(ax.get_xlim()[1], 1000)
        ax.set_xlim(lowLim, upperLim)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.title(f'cFos Density Change', fontsize=10)

        # Cleanup
        sns.despine()
        ax.set_xlabel('cFos density change (%)')
        # axes[ax_idx].set_ylabel('')

        ax.axvline(x=0, color='grey', linestyle='--', lw=0.5)
        # plt.title(f'{brainArea} - cFos Density Change', fontsize=5)
        

        # plt.savefig(f'{fileOutName}_{brainArea}_Limited.png', bbox_inches='tight')
        plt.savefig(f'{fileOutName}_{plot_idx}_Limited.png', bbox_inches='tight')
        # plt.savefig(f'{fileOutName}_{plot_idx}.png', bbox_inches='tight')
        plt.show()

### Classification based plots
def plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, titleStr, dirDict):

    conf_matrix_list_of_arrays = np.array(conf_matrix_list_of_arrays)

    print(f"{fit}: {np.mean(scores):.2f} accuracy with a standard deviation of {np.std(scores):.2f}")
    # Prepare the confusion matrix plot
    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)

    if fit != '':
        fullTitleStr = fit + ': ' + titleStr
    else:
        fullTitleStr = titleStr

    if '\n' in titleStr:
        titleStr = titleStr.replace('\n                  ', '')

    config.setup_Confmatrix_settings()

    figSizeMat = np.array(mean_of_conf_matrix_arrays.shape)/3.33
    figSizeMat[0] = figSizeMat[0] + 1
    plt.figure(figsize=figSizeMat)
    ax = sns.heatmap(mean_of_conf_matrix_arrays,cmap='Reds', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    ax.set(xticklabels=YtickLabs, yticklabels=YtickLabs, xlabel='Predicted Label', ylabel='True Label')
    # plt.title(fullTitleStr, fontsize=figSizeMat[0]*1.5)

    # Save the plot
    plt.savefig(join(dirDict['outDir_model'], f"ConfusionMatrix_{fit}"), bbox_inches='tight')     
    plt.show()

def plotPRcurve(n_classes, y_real_lab, y_prob, labelDict, Yticklabs, daObjstr, plotSwitch, fit, dirDict):
    # n_classes = int, number of classes
    # y_real, y_prob = test set labels and probabilities assigned to test set samples.
    # y_real, y_prob are in a [n_splits, n_samples, n_classes] format

    from sklearn.metrics import precision_recall_curve, auc
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.font_manager import FontProperties
    from sklearn.preprocessing import label_binarize
    
    # Convert the labels to a binary format
    y_real_lab = [label_binarize(x, classes=Yticklabs) for x in y_real_lab]
    y_real_lab = np.array(y_real_lab)
    if n_classes == 2:
        # y_real_lab = np.concatenate([y_real_lab, 1 - y_real_lab], axis=2)
        y_real_lab = np.concatenate([1 - y_real_lab, y_real_lab], axis=2)
    y_real = y_real_lab

    # Convert the arrays
    # y_real = np.array(y_real)
    y_prob = np.array(y_prob)

    y_real_all, y_prob_all = [], []

    if plotSwitch:
        figSizeMat = np.array((n_classes, n_classes))/2.2
        f = plt.figure(figsize=figSizeMat)
        # f = plt.figure()
        axes = plt.axes()

    # Depending on feature list y passed, determine if the classes are numbers or strings
    if all(isinstance(key, str) for key in labelDict.keys()):
        labelDict = {value: key for key, value in labelDict.items()}

    auc_dict = dict()

    colorDict = hf.create_color_dict(dictType='drug', rgbSwitch=0, alpha_value=0, scaleVal=False)

    for i in np.arange(n_classes):
        label_per_split = y_real[:, :, i]
        prob_per_split = y_prob[:, :, i]

        label_per_split_reshape = label_per_split.reshape(prob_per_split.size, 1)
        prob_per_split_reshape = prob_per_split.reshape(prob_per_split.size, 1)

        y_real_all.append(label_per_split_reshape)
        y_prob_all.append(prob_per_split_reshape)

        # Calculate the PR Curve
        precision, recall, _ = precision_recall_curve(label_per_split_reshape, prob_per_split_reshape)
        auc_val = auc(recall, precision)
        lab = f'{labelDict[i]}=%.2f' % (auc_val)
        auc_dict[labelDict[i]] = np.round(auc_val, 2)

        # Depending on switch, plot it.
        if plotSwitch:
            axes.step(recall, precision, label=lab, color=colorDict[labelDict[i]], lw=2)

    # Create a mean PR Curve by combining all the data
    precision, recall, _ = precision_recall_curve(np.concatenate(y_real_all), np.concatenate(y_prob_all))

    auc_val_mean = auc(recall, precision)
    lab = f'Mean Curve=%.2f' % (auc_val_mean)
    auc_dict['Mean'] = np.round(auc_val_mean, 2)

    if plotSwitch:
        axes.step(recall, precision, label=lab, lw=3, color='black')
        axes.set_xlabel('Recall')
        axes.set_ylabel('Precision')

        if n_classes == 2:
            legend = axes.legend(loc='lower left')
            # Adjust the size for purposes of the paper
            # for label in legend.get_texts():
            #     label.set_fontproperties(FontProperties(size=10, weight='bold'))
            # for label in legend.get_lines():
            #     label.set_linewidth(15)
        else:
            legend = axes.legend(loc='lower left')
            for label in legend.get_lines():
                label.set_linewidth(.5)


        plt.savefig(join(dirDict['outDir_model'], f"PRcurve_{fit}"), bbox_inches='tight')     
        plt.show()

    return auc_dict

def plot_shap_summary(X_train_trans_list, shap_values_list, y_vec, n_classes, plotDict, classifyDict, dirDict):
    import shap
    
    n_splits = len(X_train_trans_list)
    test_count = shap_values_list[0][0].shape[0]
    shap_threshold = np.ceil(n_splits * plotDict['shapSummaryThres']/100)

    X_train_trans_nonmean = pd.concat(X_train_trans_list, axis=0)
    shap_values_nonmean = []
    for shap_x_df in shap_values_list:
        if shap_x_df:
            shap_values_nonmean.append(pd.concat(shap_x_df, axis=0))

    # Plot the SHAP values for each class
    cap_shap_values = True # Cap the SHAP values at 1 and -1
    max_abs_shap_val = 1

    for shap_vals in shap_values_nonmean:
        # determine how many models across all the splits each feature was included in
        shapValueCount = shap_vals.agg(np.isnan).sum()
        feature_model_count = n_splits - shapValueCount/test_count
        svf_sorted = feature_model_count.sort_values(ascending=False)

        if plotDict['shapSummaryThres'] is not None:
            svf_sorted = svf_sorted[svf_sorted >= shap_threshold]
            maxDisp = len(svf_sorted)-1
        else:
            maxDisp = plotDict['shapMaxDisplay']

        # For effective plotting and sorting purposes, NaNs -> 0s
        shap_vals = shap_vals.fillna(0)

        # Filter the data for the top features
        sortingIdx = svf_sorted.index
        X_train_trans_sorted = X_train_trans_nonmean[sortingIdx]
        shap_vals = shap_vals[sortingIdx]

        # if there are 2 classes, sort by the median difference, by use of the index which determines the true label
        if n_classes == 2:
            # Map the index column to a new 'drug' column using y_vec
            shap_vals['drug'] = y_vec[shap_vals['index']]

            # Identify unique drug names
            drugList = list(shap_vals['drug'].unique())

            drugMedians = pd.DataFrame(index=list(shap_vals.columns[1:-1]), columns=drugList)
            shap_vals.reset_index(inplace=True, drop=True)
            for drug in drugList:
                drugMedians.loc[:, drug] = shap_vals.loc[shap_vals['drug'] == drug].median()
            
            # Find the difference between the medians
            drugMedians['medianDiff'] = abs(drugMedians[drugList[0]] - drugMedians[drugList[1]])

            # Sort by the median difference
            drugMedians_sort = drugMedians.sort_values(by='medianDiff', ascending=False)

            # Resort data by the drug median
            shap_vals = shap_vals[drugMedians_sort.index]
            X_train_trans_sorted = X_train_trans_sorted[drugMedians_sort.index]
        else:
            # Discard the index column. No such sorting implemented atm.
            shap_vals = shap_vals.drop('index', axis=1)
            X_train_trans_sorted = X_train_trans_sorted.drop('index', axis=1)

        sortSHAP = False
        parenVal = 'medianDiff'

        if parenVal == 'count' or n_classes > 2:
            # Adjust the feature names to include their counts.
            parenVal = [int(x) for x in svf_sorted.values[1:]]
            featureNames = [f"{feat} ({int(svf_sorted[feat])})" for feat in list(shap_vals.columns)]
        elif parenVal == 'medianDiff':
            parenVal = drugMedians_sort.medianDiff.round(2)
            featureNames = [f"{feat} ({parenVal[feat]})" for feat in list(shap_vals.columns)]

        if cap_shap_values:
            shap_vals = shap_vals.clip(lower=-max_abs_shap_val, upper=max_abs_shap_val)

        # Find the min and the max of the SHAP values
        shap_values_min = shap_vals.min().min()
        shap_values_max = shap_vals.max().max()

        # Find whether the min or max is larger in magnitude and assgin it to 'maxVal'
        if abs(shap_values_min) > shap_values_max:
            maxVal = abs(shap_values_min)
        else:
            maxVal = shap_values_max
        maxVal = maxVal + 0.01

        # Create a method that accounts for the number of features in the plot to set the figure size. Scale for purposes of other elements.
        scaleFactor = 3
        figW = 1.763
        figH = 0.141 + (0.09 * len(shap_vals.columns)) # Original height per data 0.0913125
        # figSize = (figH * scaleFactor, figW * scaleFactor)
               
        # Plot the SHAP values
        if 'MDMA' in classifyDict['label']:

            firstHalfIdx = int(len(featureNames)/2)
            shap.summary_plot(shap_vals.values[:, 0:firstHalfIdx], X_train_trans_sorted.values[:, 0:firstHalfIdx], feature_names=featureNames[0:firstHalfIdx], sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score', plot_size=.3)
            plt.xlim([-maxVal, maxVal])
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_1_Sym.svg"), format='svg', bbox_inches='tight')
            plt.show()

            shap.summary_plot(shap_vals.values[:, firstHalfIdx:], X_train_trans_sorted.values[:, firstHalfIdx:], feature_names=featureNames[firstHalfIdx:], sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score', plot_size=.3)
            plt.xlim([-maxVal, maxVal])
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_2_Sym.svg"), format='svg', bbox_inches='tight')
            plt.show()
        else:
            shap.summary_plot(shap_vals.values, X_train_trans_sorted.values, feature_names=featureNames, sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score', plot_size=.45)
            plt.xlim([-maxVal, maxVal])
            # Save the plot +/- Titling it.
            # Update the font on the x axis tick labels

            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_Sym.svg"), format='svg', bbox_inches='tight') #, bbox_inches='tight'
            plt.show()

def plot_shap_bar(explainers, X_train_trans_list, shap_values_list, y_vec, n_classes, plotDict, classifyDict, dirDict):
    import shap

    n_splits = len(X_train_trans_list)
    shap_threshold = np.ceil(n_splits * plotDict['shapSummaryThres']/100)

    X_train_trans_nonmean = pd.concat(X_train_trans_list, axis=0)
    shap_values_nonmean = []
    if n_classes == 2:
        shap_values_nonmean.append(pd.concat(shap_values_list, axis=0))
        test_count = shap_values_list[0].shape[0]
    else:
        test_count = shap_values_list[0][0].shape[0]
        for shap_x_df in shap_values_list:
            shap_values_nonmean.append(pd.concat(shap_x_df, axis=0))

    # Plot the SHAP values for each class
    cap_shap_values = True # Cap the SHAP values at 1 and -1
    max_abs_shap_val = 1

    # expOut = []
    # for explainer, x_data in zip(explainers, X_train_trans_list):
    #     expOut1 = explainer(x_data.drop('index', axis=1))



    for shap_vals in shap_values_nonmean:
        # determine how many models across all the splits each feature was included in
        shapValueCount = shap_vals.agg(np.isnan).sum()
        feature_model_count = n_splits - shapValueCount/test_count
        svf_sorted = feature_model_count.sort_values(ascending=False)

        if plotDict['shapSummaryThres'] is not None:
            svf_sorted = svf_sorted[svf_sorted >= shap_threshold]
            maxDisp = len(svf_sorted)-1
        else:
            maxDisp = plotDict['shapMaxDisplay']

        # For effective plotting and sorting purposes, NaNs -> 0s
        shap_vals = shap_vals.fillna(0)

        # Filter the data for the top features
        sortingIdx = svf_sorted.index
        X_train_trans_sorted = X_train_trans_nonmean[sortingIdx]
        shap_vals = shap_vals[sortingIdx]

        # if there are 2 classes, sort by the median difference
        if n_classes == 2:
            # Map the index column to a new 'drug' column using y_vec
            shap_vals['drug'] = y_vec[shap_vals['index']]

            # Identify unique drug names
            drugList = list(shap_vals['drug'].unique())

            drugMedians = pd.DataFrame(index=list(shap_vals.columns[1:-1]), columns=drugList)
            shap_vals.reset_index(inplace=True, drop=True)
            for drug in drugList:
                drugMedians.loc[:, drug] = shap_vals.loc[shap_vals['drug'] == drug].median()
            
            # Find the difference between the medians
            drugMedians['medianDiff'] = abs(drugMedians[drugList[0]] - drugMedians[drugList[1]])

            # Sort by the median difference
            drugMedians_sort = drugMedians.sort_values(by='medianDiff', ascending=False)

            # Resort data by the drug median
            shap_vals = shap_vals[drugMedians_sort.index]
            X_train_trans_sorted = X_train_trans_sorted[drugMedians_sort.index]

        sortSHAP = True
        parenVal = 'medianDiff'

        # explainer(x_data.drop('index', axis=1))
        thingOut = explainers[0](X_train_trans_list[0].drop('index', axis=1))
        shap.plots.bar(thingOut, max_display=12)
        plt.show()


        if parenVal == 'count':
            # Adjust the feature names to include their counts.
            parenVal = [int(x) for x in svf_sorted.values[1:]]
            featureNames = [f"{feat} ({svf_sorted[feat]})" for feat in list(shap_vals.columns)]
        elif parenVal == 'medianDiff':
            parenVal = drugMedians_sort.medianDiff.round(2)
            featureNames = [f"{feat} ({parenVal[feat]})" for feat in list(shap_vals.columns)]

        if cap_shap_values:
            shap_vals = shap_vals.clip(lower=-max_abs_shap_val, upper=max_abs_shap_val)

        # Find the min and the max of the SHAP values
        shap_values_min = shap_vals.min().min()
        shap_values_max = shap_vals.max().max()

        # Find whether the min or max is larger in magnitude and assgin it to 'maxVal'
        if abs(shap_values_min) > shap_values_max:
            maxVal = abs(shap_values_min)
        else:
            maxVal = shap_values_max
        maxVal = maxVal + 0.01

        # Create a method that accounts for the number of features in the plot to set the figure size. Scale for purposes of other elements.
        scaleFactor = 5
        figW = 1.763
        figH = 0.141 + (0.0913125 * len(shap_vals.columns)) * 1.5
        figSize = (figW * scaleFactor, figH * scaleFactor)
        #, plot_size=figSize
               
        # Plot the SHAP values
        if 'MDMA' in classifyDict['label']:
            firstHalfIdx = int(len(featureNames)/2)
            pltHdl = shap.summary_plot(shap_vals.values[:, 0:firstHalfIdx], X_train_trans_sorted.values[:, 0:firstHalfIdx], feature_names=featureNames[0:firstHalfIdx], sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score')
            plt.xlim([-maxVal, maxVal])
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_1_Sym.svg"), format='svg', bbox_inches='tight')
            plt.show()

            shap.summary_plot(shap_vals.values[:, firstHalfIdx:], X_train_trans_sorted.values[:, firstHalfIdx:], feature_names=featureNames[firstHalfIdx:], sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score')
            plt.xlim([-maxVal, maxVal])
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_2_Sym.svg"), format='svg', bbox_inches='tight')
            plt.show()
        else:
            pltHdl = shap.summary_plot(shap_vals.values, X_train_trans_sorted.values, feature_names=featureNames, sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score')
            plt.xticks(fontsize=14)
            plt.xlim([-maxVal, maxVal])
            # Save the plot +/- Titling it.
            # Update the font on the x axis tick labels

            # shap.plots.bar(shap_vals.values, max_display=12)
            # plt.show()

            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_Sym.svg"), format='svg', bbox_inches='tight') #, bbox_inches='tight'
            plt.show()

def plot_shap_force(X_train_trans_list, shap_values_list, baseline_val, y_real_lab, numYDict, plotDict, dirDict):
    import itertools
    import shap

    # Set font to 12 pt Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.size'] = 5
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5

    if plotDict['shapForcePlotCount'] == 0:
        return
    else:
        forcePlotCount = plotDict['shapForcePlotCount']

    class_count = len(numYDict.keys())
    n_splits = len(X_train_trans_list)
    test_count = shap_values_list[0].shape[0]

    # Generate an array of true labels
    labelDict = {value: key for key, value in numYDict.items()}

    regionList = True
    regionListSet = ['VISpm', 'LH', 'PF']

    if regionList:
        cvSplit1, testPoint1, cvSplit2, testPoint2 = find_max_min_index(shap_values_list, regionListSet)
        cvSplitSet = [cvSplit1] * 4
        testPointSet = [0, 1, 2, 3]
        cvSplitTest = list(zip(cvSplitSet, testPointSet))
    else:
        cvSplitTest = np.random.randint(np.zeros((1,forcePlotCount)), [[n_splits], [test_count-1]], dtype=np.uint8).T

    idx_db = sortShap(shap_values_list, regionListSet)
    idx_db = list(idx_db.sort_values(by='normVal', ascending=False).index)

    # Replicate each of the points in idx_db by 4
    idx_db2 = np.tile(idx_db, [4, 1]).T

    # Reshape the list into a 1D array
    idx_db2 = idx_db2.reshape(-1)
    testPointSet = [0, 1, 2, 3] * 100

    cvSplitTest = list(zip(idx_db2, testPointSet))

    cvSplitTest = cvSplitTest[0:forcePlotCount]

    # Do an example force plot
    if class_count == 2:

        for cvSplit, testPoint in cvSplitTest:
            
            y_labels = y_real_lab[cvSplit]
            idStr = f"CV{cvSplit}_Sample{testPoint}"
            titleStr = f'Test Sample of {y_labels[testPoint]}, {idStr}'

            # Extract some key variables
            featCount = shap_values_list[cvSplit].shape[1]
            shapVals = np.round(shap_values_list[cvSplit].iloc[testPoint,1:featCount].values, 2)
            testVals = np.round(X_train_trans_list[cvSplit].iloc[testPoint,1:featCount].values, 2)
            featNames = list(shap_values_list[cvSplit].columns[1:featCount])

            # Plot
            shap.plots.force(baseline_val[cvSplit], shap_values=shapVals, features=testVals, feature_names=featNames, out_names=None, link='identity', plot_cmap='RdBu', matplotlib=True, figsize=(14, 2), show = False)

            # Modify the plot
            plt.title(titleStr, y=1.5)

            # Save the plot
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_example_{idStr}.svg"), format='svg', bbox_inches='tight')
            plt.show()

def plot_feature_scores(clf, featureNames):
    # A function which plots feature scores from a pipeline object if that feature selection method has scores/pvalues.
    # Examples - Fdr, Fwe, and Fwe_BH

    support_ = clf['featureSel'].get_support()
    scores_ = clf['featureSel'].scores_[support_]
    pvalues_ = clf['featureSel'].pvalues_[support_]
    
    # Sort arrays based on pvalues_
    sort_indices = np.argsort(pvalues_)
    sorted_score = scores_[sort_indices]
    sorted_pvalues = pvalues_[sort_indices]
    sorted_featureNames = [featureNames[i] for i in sort_indices]

    # Create the horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot horizontal bars
    bars = ax.barh(range(len(sorted_score)), sorted_score)

    # Add pvalues as text on top of each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{sorted_pvalues[i]:.2e}", ha='center', va='center')

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(sorted_score)))
    ax.set_yticklabels(sorted_featureNames)

    plt.tight_layout()
    plt.show()

def plot_histogram(data, dirDict):
    plt.title('Feature Count per CV', fontdict={'fontsize': 20})
    plt.hist(data, bins=10, edgecolor='black')
    plt.savefig(os.sep.join([dirDict['outDir_model'], "featureCountHist.svg"]), format='svg', bbox_inches='tight')          
    plt.show()

def plot_cross_model_AUC(scoreNames, aucScores, aucScrambleScores, dirDict):
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'

    # Color palette
    colorsList = [[100, 100, 100], [180, 180, 180]]
    colorsList = np.array(colorsList)/256

    plt.figure(figsize=(1.9, 1.9))  # Adjust the width and height as needed

    plt.barh(scoreNames, aucScores, label='Data', color=colorsList[0])
    plt.barh(scoreNames, aucScrambleScores, label='Shuffled', color=colorsList[1])

    # Set labels and title,
    plt.xlabel('Mean Precision-Recall AUC')

    for index, value in enumerate(aucScores):
        percentage_text = '{:.0%}'.format(value)  # Format the value as a percentage
        plt.text(value-.01, index, percentage_text, ha='right', va='center', color='white')

    for index, value in enumerate(aucScrambleScores):
        percentage_text = '{:.0%}'.format(value)  # Format the value as a percentage
        plt.text(value-.01, index, percentage_text, ha='right', va='center')

    # Display the plot
    plt.savefig(os.sep.join([dirDict['crossComp_figDir'], f"MeanAUC_barplot.svg"]), format='svg', bbox_inches='tight')
    plt.show()

def plot_cross_model_Accuracy(scoreNames, meanScores, meanScrambleScores, colorsList, saveDir):
    import matplotlib.pyplot as plt

    # Plot the bar chart
    plt.figure(figsize=(5, 5))  # Adjust the width and height as needed
    plt.barh(scoreNames, meanScores, label='Data', color=colorsList[0])
    plt.barh(scoreNames, meanScrambleScores, label='Shuffled', color=colorsList[1])

    # Add a legend explaining colors
    plt.legend()

    # Set labels and title
    plt.xlabel('Score')
    plt.ylabel('Classification')
    plt.title('Mean Accuracy across cross-validation')

    # Add percentage text to the plot
    for index, value in enumerate(meanScores):
        percentage_text = '{:.0%}'.format(value)  # Format the value as a percentage
        plt.text(value-.01, index, percentage_text, ha='right', va='center', weight='bold', fontsize=10)

    for index, value in enumerate(meanScrambleScores):
        percentage_text = '{:.0%}'.format(value)  # Format the value as a percentage
        plt.text(value-.01, index, percentage_text, ha='right', va='center', weight='bold', fontsize=10)

    # Save and show the plot
    plt.savefig(os.sep.join([saveDir ,"MeanAcc_barplot.svg"]), format='svg', bbox_inches='tight')     
    plt.show()

def find_max_index(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    max_index = None  # Initialize max_index to store the corresponding index in shap_values_list
    max_row_index = None  # Initialize max_row_index to store the index of the row with the maximum values

    for idx, shap_val_tab in enumerate(shap_values_list):
        # Check if all elements in regionSet are present in the DataFrame
        if not all(region in shap_val_tab.columns for region in regionSet):
            # Skip to the next iteration if not all elements are present
            continue

        # Calculate the normalized values for 'VISpm' and 'LH'
        normalized_values = shap_val_tab.loc[:, regionSet].abs() / np.abs()

        # Calculate the sum of 'VISpm' and 'LH' for each row
        row_sums = normalized_values.sum(axis=1)

        # Find the index with the maximum sum
        current_max_value = row_sums.max()
        if current_max_value > max_value:
            max_value = current_max_value
            max_index = idx
            max_row_index = row_sums.idxmax()

    return max_index, max_row_index

def find_max_min_index(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    min_value = float('inf')  # Initialize min_value to store the minimum value
    max_index = None  # Initialize max_index to store the corresponding index in shap_values_list
    min_index = None  # Initialize min_index to store the corresponding index in shap_values_list
    max_row_index = None  # Initialize max_row_index to store the index of the row with the maximum values
    min_row_index = None  # Initialize min_row_index to store the index of the row with the minimum values

    idxList = []

    for idx, shap_val_tab in enumerate(shap_values_list):
        # Check if all elements in regionSet are present in the DataFrame
        if not all(region in shap_val_tab.columns for region in regionSet):
            # Skip to the next iteration if not all elements are present
            continue

        # Calculate the normalized values for the specified regions in regionSet
        normalized_values = shap_val_tab.loc[:, regionSet] / shap_val_tab.iloc[:, 1:].max().max()

        # Calculate the sum of the specified regions for each row
        row_sums = normalized_values.sum(axis=1)

        # Find the index with the maximum sum
        current_max_value = row_sums.max()
        if current_max_value > max_value:
            max_value = current_max_value
            max_index = idx
            max_row_index = row_sums.idxmax()
            idxList.append(idx)

        # Find the index with the minimum sum
        current_min_value = row_sums.min()
        if current_min_value < min_value:
            min_value = current_min_value
            min_index = idx
            min_row_index = row_sums.idxmin()

    max_index = idxList[-1]

    return max_index, max_row_index, min_index, min_row_index

def plot_featureCount_violin(scoreNames, featureLists, dirDict):
    import seaborn as sns
    import pandas as pd

    colorsList = [[82, 211, 216], [56, 135, 190]]
    colorsList = np.array(colorsList)/256

    # Your list of lists (sublists with numbers)
    data = [[len(sublist) for sublist in inner_list] for inner_list in featureLists]

    # Reverse the order of the data and scoreNames
    data = data[::-1]
    scoreNames = scoreNames[::-1]

    df = pd.melt(pd.DataFrame(data, index=scoreNames).T, var_name='Category', value_name='Values')

    # Create horizontally oriented violin plot
    plt.figure(figsize=(2, 1.95))  # Adjust the width and height as needed

    ax = sns.violinplot(x='Values', y='Category', data=df, orient='h', color=colorsList[0], linewidth=0.5)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    ax.xaxis.set_tick_params(length=2, width=1)
    ax.yaxis.set_tick_params(length=3, width=1)

    # Set plot labels and title
    plt.xlabel('Brain regions used by classifier')
    plt.ylabel('')

    plt.savefig(os.sep.join([dirDict['crossComp_figDir'], "RegionCountPerSplit_violin.svg"]), format='svg', bbox_inches='tight')     

    # Show the plot
    plt.show()

def plot_similarity_matrix(scoreNames, featureLists, filterByFreq, dirDict):
    from matplotlib import cm

    modelCount = len(featureLists)
    # regionDict = dict(Counter(featureLists[0]))
    # labels, counts = list(regionDict.keys()), list(regionDict.values())

    # Initialize a grid
    grid = [[0 for _ in range(modelCount)] for _ in range(modelCount)]

    # Flatten every list
    featureListFlat = [[element for item in subList for element in item] for subList in featureLists]

    jacSim = False 

    # compare the mean distances across items of the list
    for idx_a, listA in enumerate(featureListFlat):
        for idx_b, listB in enumerate(featureListFlat):
            
            if jacSim:
                # Jaccard Sim
                grid[idx_a][idx_b] = hf.weighted_jaccard_similarity(listA, listB, 75)
            else:
                # Overlap count
                _, _, intersection = hf.overlapCounter(listA, listB, filterByFreq)
                grid[idx_a][idx_b] = len(intersection)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(7,7))
    im = sns.heatmap(grid, cmap='Blues', annot=True, fmt='.0f', ax=ax, yticklabels=scoreNames, xticklabels=scoreNames, annot_kws={'size': 15})

    # Remove the colorbar
    cbar = ax.collections[0].colorbar
    cbar.remove()

    # Set font size for x-axis ticks and labels
    ax.tick_params(axis='x', labelsize=12)

    # Set font size for y-axis ticks and labels
    ax.tick_params(axis='y', labelsize=12, rotation=0)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Cycle through sns heatmap annotations and remove those that are equal to 0
    for text in ax.texts:
        if int(text.get_text()) == 0:
            text.set_text("")

    # plt.title(titleStr, fontdict={'fontsize': 18})
    plt.savefig(os.sep.join([dirDict['crossComp_figDir'], "MeanSimilarity_heatmap.svg"]), format='svg', bbox_inches='tight')     
    plt.show()

def plot_featureOverlap_VennDiagram(scoreNames, featureLists, filterByFreq, dirDict):
    wrapper = textwrap.TextWrapper(width=12, break_on_hyphens=False)  # Adjust width as needed

    # Flatten every list
    featureListFlat = [[element for item in subList for element in item] for subList in featureLists]

    # Sample data
    for idx1, list1 in enumerate(featureListFlat):
        for idx2, list2 in enumerate(featureListFlat):

            # Skip the same list
            if idx1 == idx2:
                continue

            # Filter out features in each counter whose count is not above it.
            only_list1, only_list2, intersection = hf.overlapCounter(list1, list2, filterByFreq)

            # Skip if there is no overlap
            if intersection == []:
                continue

            # Create a Venn diagram
            venn_diagram = venn2(subsets=(len(only_list1), len(only_list2), len(intersection)/2),
                                set_labels=(scoreNames[idx1], scoreNames[idx2]))


            venn_labels = {'100': only_list1, '010': only_list2, '110': intersection}
            for idx, (labId, labels) in enumerate(venn_labels.items()):
                wrapped_labels = wrapper.fill(text='  '.join(labels))
                venn_diagram.get_label_by_id(labId).set_text(wrapped_labels)
                venn_diagram.get_label_by_id(labId).set_fontsize(8)  # Adjust font size if needed

            # # Customize the size of the Venn diagram
            # plt.gcf().set_size_inches(8, 8)
            figName = f'VD_{scoreNames[idx1]} and {scoreNames[idx2]}'
            figName = figName.replace('/', '+')
            figName = figName.replace(' ', '_')
            plt.savefig(os.sep.join([dirDict['crossComp_figDir'], f"{figName}.svg"]), format='svg', bbox_inches='tight')     

            # Display the plot
            plt.show()

def plot_featureHeatMap(df_raw, scoreNames, featureLists, filterByFreq, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # Creates the heatmap for the data

    scoreNames = scoreNames[::-1]
    featureLists = featureLists[::-1]

    # Set variables
    dataFeature = 'abbreviation'
    plt.rcParams['font.size'] = 6
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    sys.path.append('../dependencies/')

    # Create a sorted structure from the data for scaffolding the desired heatmap
    brainAreaColorDict = hf.create_color_dict(dictType='brainArea', rgbSwitch=0)
    regionArea = hf.create_region_to_area_dict(df_raw, 'abbreviation')
    regionArea['Region_Color'] = regionArea['Brain_Area'].map(brainAreaColorDict)

    df_Tilted = df_raw.pivot(index='abbreviation', columns='dataset', values='count')
    df_Tilted = df_Tilted.reindex(regionArea['abbreviation'].tolist(), axis=0)

    featureListFlat = [[element for item in subList for element in item] for subList in featureLists]

    # Process the data from above
    featureListDicts = [hf.listToCounterFilt(x, filterByFreq=0) for x in featureListFlat]
    featureListArray = [list(x.keys()) for x in featureListDicts]

    # Add in columns for each of the actual comparisons.
    df_frame = df_Tilted.reindex(columns=scoreNames)
    df_frame.fillna(0, inplace=True)
        
    for idx, (comp, featureList) in enumerate(zip(df_frame.columns, featureListDicts)):
        for regionName in featureList.keys():
            df_frame.loc[regionName, comp] = featureList[regionName]
            
    # Remove any rows which are not above threshold
    df_plot = df_frame[df_frame.sum(axis=1) >= filterByFreq]
        
    # Remove the abbreviations from regionArea not represented in df_plot, Filter the regionArea for 'Cortex' and 'Thalamus'
    regionArea = hf.create_region_to_area_dict(df_raw, dataFeature)
    regionArea = regionArea[regionArea['abbreviation'].isin(df_plot.index)]
    regionArea = regionArea[regionArea['Brain_Area'].isin(['Cortex', 'Thalamus'])]

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]
    modelCount = len(df_plot.columns)

    # merge df_plot and regionArea, moving the Brain_Area_Idx and Brain_Area columns to df_plot
    df_plot_combo = df_plot.merge(regionArea, left_index=True, right_on=dataFeature)

    # Cycle through the df_plot_combo's distinct Brain_Area_Idx, and resort data by row sums
    newIdx = []
    for idx in regionArea.Brain_Area_Idx.unique():
        # Identify which regions have the same Brain_Area_Idx
        df_seg = df_plot_combo[df_plot_combo.Brain_Area_Idx == idx]
        sorted_seg_idx = df_seg.iloc[:, 0:modelCount].sum(axis=1).sort_values(ascending=False).index

        # Append to list
        newIdx = newIdx + list(sorted_seg_idx)

    # Resort the data
    df_plot = df_plot_combo.reindex(newIdx, axis=0)
    df_plot = df_plot.set_index('abbreviation')
    df_plot = df_plot.drop(columns=['Brain_Area_Idx'])

    # Drop the columns which do not include the string 'PSI'
    df_plot = df_plot.loc[:, df_plot.columns.str.contains('PSI') | (df_plot.columns == 'Brain_Area')]

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    formatter.set_powerlimits((-2, 2))

    colorbar = [False, True]
    axes = []
    regionSet = ['Cortex', 'Thalamus']

    for idx, regionName in enumerate(regionSet):

        # Slice and modify previous structures to create segment
        df_plot_seg = df_plot[df_plot.Brain_Area == regionName].drop(columns=['Brain_Area'])

        # Sort by highest sum of row
        df_plot_seg = df_plot_seg.loc[df_plot_seg.sum(axis=1).sort_values(ascending=False).index]

        matrix = df_plot_seg.values

        xticklabels = df_plot_seg.columns.values.tolist()
        yticklabels = df_plot_seg.index.values.tolist()

        figwidth = len(xticklabels)*0.1433
        figheight = len(yticklabels)*0.1433

        f = plt.figure(figsize=(figwidth, figheight))  # Adjust the width and height as needed
        ax = f.add_subplot(111)
        axes.append(ax)

        heatmap = sns.heatmap(matrix, cmap='crest', ax=axes[idx] , fmt='.2f', cbar = False, square=True, yticklabels=yticklabels, xticklabels=xticklabels, cbar_kws={"format": formatter}, center=0, linewidths=0.5, linecolor='black')
        axes[idx].tick_params(left=True, bottom=True, width=0.5, length=2)

        # Rotate the xticklabels 45 degrees
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # if cbs:
        #     cbar = heatmap.figure.colorbar(heatmap.collections[0], ax=axes[idx], location='right', use_gridspec=True, pad=0.05)
        #     cbar.set_label('Feature Count', rotation=270, labelpad=5)
        #     cbar.ax.yaxis.set_major_formatter(formatter)

        plt.savefig(os.sep.join([dirDict['crossComp_figDir'], f"FeatureCountHeatmap_{regionName}.svg"]), format='svg', bbox_inches='tight')
        plt.show()

def sortShap(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    min_value = float('inf')  # Initialize min_value to store the minimum value
    idxList = []
    databaseIdx = pd.DataFrame(index=np.arange(0, len(shap_values_list)), columns=['normVal'])

    for idx, shap_val_tab in enumerate(shap_values_list):
        # Check if all elements in regionSet are present in the DataFrame
        if not all(region in shap_val_tab.columns for region in regionSet):
            # Skip to the next iteration if not all elements are present
            continue

        # Calculate the normalized values for the specified regions in regionSet
        normalized_values = shap_val_tab.loc[:, regionSet] / shap_val_tab.iloc[:, 1:].max().max()

        # Calculate the sum of the specified regions for each row
        row_sums = normalized_values.sum(axis=1)

        databaseIdx.loc[idx, 'normVal'] = row_sums.max()
        # Find the index with the maximum sum
        current_max_value = row_sums.max()
        if current_max_value > max_value:
            max_value = current_max_value
            max_index = idx
            max_row_index = row_sums.idxmax()
            idxList.append(idx)

        # Find the index with the minimum sum
        current_min_value = row_sums.min()
        if current_min_value < min_value:
            min_value = current_min_value

    return databaseIdx

def genereate_cFos_gene_corr_plots(geneDict, geneColorDict, setNames, regionSet, plotNameDict, dirDict):

    unique_genes = list(set(hf.flatten(geneDict.values())))
    genePlotColorPalette = sns.color_palette("Spectral", as_cmap=True, n_colors=len(unique_genes))

    rows = len(geneDict)
    cols = np.max(list({key: len(value) for key, value in geneDict.items()}.values()))

    for set_i, set_name in enumerate(regionSet):

        fig, axs = plt.subplots(rows, cols, figsize=(10, len(geneDict.keys())*1.67))

        for drug_i, drug in enumerate(geneDict.keys()):

            genePlotList = geneDict[drug]
            
            for geneSet_i, genePlotList_data in enumerate(genePlotList):
            
                #plot the distribution for all the gene correlations
                plt.figure(figsize=(2,1))
                sns.set(style="ticks")
                drug_data = pd.read_pickle(os.path.join(dirDict['geneCorrDir'], f'{drug}_{set_name}_corr_db.h5'))
                axHand = axs[drug_i, geneSet_i]

                ax = sns.histplot(data=drug_data , x = drug + " correlation", element = 'step', fill = False, color='grey', ax=axHand) #, lw=7
                sns.despine()

                trans = ax.get_xaxis_transform()

                # Plot the individual
                genes_of_interest = drug_data[drug_data['gene'].isin(genePlotList_data)]
                corrData = list(genes_of_interest[drug + ' correlation'])

                genes_of_interest['colorInd'] = [geneColorDict[drug] for drug in genes_of_interest.gene]
                genes_of_interest = genes_of_interest.sort_values(drug + ' correlation')

                # Generate spots for the text above the plots to go. Move text over if it is overlapping with text to the left
            
                textXVals = np.array(genes_of_interest[drug + ' correlation'])
                minDist = .03

                xLimits = ax.get_xlim()
                xLimLeftLabel = textXVals[0] 
                xLimLeftLabel = xLimits[0]
                if abs(textXVals[0]-textXVals[-1]) < 0.2:
                    xLimRightLabel = textXVals[-1] + .2
                else:
                    xLimRightLabel = textXVals[-1] #xLimits[0]+(xLimits[1]*.7)

                geneCount = len(genePlotList_data)
                textXAxes = np.linspace(xLimLeftLabel, xLimLeftLabel+(geneCount*0.1), num:=geneCount)
                # textXAxes = np.linspace(xLimLeftLabel, xLimLeftLabel+(geneCount*0.05), num:=geneCount)

                plt.sca(axHand)
                plt.xlim(-0.85, 0.85)
                plt.ylim(0, 1050)
                y_pos = 0.90

                # For plotting, flip the order of genes.
                genes_of_interest = genes_of_interest.sort_values('percentile', ascending=False)

                config.setup_mRNA_corr_settings()

                # Iterate across genes of interest to plot lines and text
                for gene_i, gene in enumerate(genes_of_interest.gene):
                    
                    geneData = genes_of_interest[genes_of_interest.gene == gene]
                    corrVal = float(geneData[drug + ' correlation'])
                    prcVal = round(float(geneData['percentile'])*100)
                    lineCol = genePlotColorPalette(geneData.colorInd)[0]

                    # Draw lines and text
                    plt.axvline(x=corrVal, color=lineCol, linestyle='--') #, lw=10
                    plt.text(-0.83, y_pos, f'{gene} ({str(prcVal)}%)', transform=trans, color=lineCol, fontsize=8) #, rotation=45
                    
                    y_pos -= 0.15

                # plotLineWidth = 5

                if drug_i == len(geneDict.keys())-1:
                    plt.xlabel('Correlation', labelpad=1)
                else:
                    plt.xlabel('')

                if geneSet_i == 0:
                    plt.ylabel('Number of genes', labelpad=.5)
                else:
                    plt.ylabel('')

                plotTitle = f'{plotNameDict[drug]} vs {setNames[geneSet_i]} ({set_name})'
                plt.title(plotTitle, loc='center', fontsize=8, pad=10) #pad=300
                plt.subplots_adjust(hspace=0.6, wspace=0.2) 

        plt.savefig(os.path.join(dirDict['outDir'], f'{set_name}_Drug_vs_Receptors.svg'), bbox_inches='tight')

        plt.show()
        plt.clf()