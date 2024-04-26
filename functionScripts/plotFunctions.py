import os, sys, shap
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

sys.path.append('dependencies')



def plotTotalPerDrug(pandasdf, column2Plot, dirDict, outputFormat):
    totalCellCountData = pandasdf[pandasdf.Region_ID == 88]

    # plotting
    # sns.set(font_scale=3)
    sns.set_style('ticks')
    sns.despine()

    # Shift the color codes to RGB and add alpha
    colorDict = hf.create_color_dict('drug', 0)
    # boxprops = dict(alpha=0.7)
    boxprops = dict()

    plt.figure(figsize=(9.5, 3.5))
    ax = sns.boxplot(x="drug", y=column2Plot, data=totalCellCountData, whis=0, dodge=False, showfliers=False, linewidth=.5, hue='drug', palette=colorDict, boxprops=boxprops)
    sns.scatterplot(x="drug", y=column2Plot, data=totalCellCountData, hue='drug', linewidth=0, style='sex', markers=True, s=50, palette=colorDict, ax=ax, edgecolor='black')

    # remove legend
    plt.legend([], [], frameon=False)

    # cleanup
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.xaxis.set_tick_params(length=2, width=1)
    ax.yaxis.set_tick_params(length=2, width=1)
    ax.set_xlabel('Drug', fontdict={'fontsize':15})
    ax.set_ylabel('Total Cells (Count)', fontdict={'fontsize':20})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)

    ax.set_yscale('log')
    # ax.set(ylim=(8e5, 1e7))
    sns.despine()

    plt.savefig(dirDict['outDir'] + 'totalCells.' + outputFormat,  format=outputFormat, bbox_inches='tight')

    sns.set_theme()

def plotLowDimEmbed(pandasdf, column2Plot, dirDict, dimRedMeth, classifyDict):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler, PowerTransformer
    from itertools import combinations
    from sklearn.pipeline import Pipeline

    colorHex = hf.create_color_dict(dictType='drug')

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

    n_comp = 4

    # Apply some preprocessing to mimic pipeline
    transMod = PowerTransformer(method='yeo-johnson', standardize=False)
    scaleMod = RobustScaler()

    if dimRedMeth == 'PCA':
        dimRedMod = PCA(n_components=n_comp)
    elif dimRedMeth == 'LDA':
        dimRedMod = LDA(n_components=n_comp)
    else:
        KeyError('dimRedMethod not recognized, pick LDA or PCA')

    pipelineList = [('transMod', transMod), ('scaleMod', scaleMod), ('dimRedMod', dimRedMod)]
    # pipelineList = [('scaleMod', scaleMod), ('dimRedMod', dimRedMod)]
    pipelineObj = Pipeline(pipelineList)

    # Scale the data
    df_Tilted_transformed = pipelineObj.fit_transform(df_Tilted.iloc[:, :-1], df_Tilted.iloc[:, -1])
    customOrder = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']

    # Create average cases for each drug.
    compName = dimRedMeth[0:2]
    colNames = [f"{compName}{x}" for x in range(1, n_comp+1)]

    dimRedData = pd.DataFrame(data=df_Tilted_transformed, index=df_Tilted.index, columns=colNames)
    dimRedData.loc[:, 'drug'] = pd.Categorical(y, categories=customOrder, ordered=True)
    dimRedDrugMean = dimRedData.groupby(by='drug').mean()

    # Means aren't sorted like the centers. Problems here are clear when the mean dot isn't the same color.
    resortIdx = [1, 2, 3, 0, 4, 5, 6, 7]
    dimRedDrugMean = dimRedDrugMean.iloc[resortIdx]

    # Plot
    sns.set(font_scale=2)
    sns.set_style('ticks')
    sns.despine()

    pairs = list(combinations(range(n_comp), 2))

    for comp_pair in pairs:
        col1 = colNames[comp_pair[0]]
        col2 = colNames[comp_pair[1]]

        plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
        sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData, s=50, alpha=0.75, palette=colorHex)
        sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean, s=100, legend=False, edgecolor='black', palette=colorHex)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)

        # Customize the plot
        # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
        plt.title(f"Linear Discrimants of Normalized counts", fontsize=20)
        plt.xlabel(col1, fontsize=20)
        plt.xticks(fontsize=15)
        plt.ylabel(col2, fontsize=20)
        plt.yticks(fontsize=15)

        # Save
        plt.savefig(dirDict['outDir'] + f"dimRed_{filtTag}_{col1} x {col2}.{dirDict['outputFormat']}", format=dirDict['outputFormat'], bbox_inches='tight')

        plt.show()
    
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

        plt.savefig(os.path.join(outDirPath, f"scaleChain_{feat}.{dirDict['outputFormat']}"), format=dirDict['outputFormat'], bbox_inches='tight')

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

        plt.savefig(dirDict['outDir'] + f'Dist_{dist_met}_raw.png', format='png', bbox_inches='tight')

        sns.clustermap(pairwise, 
                       cmap='rocket', 
                       fmt='.2f', 
                        dendrogram_ratio = 0.1)

        plt.title(dist_met, fontsize=45)
        plt.savefig(dirDict['outDir'] + f'Dist_{dist_met}_clustered.png', format='png', bbox_inches='tight')

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
        
        plt.savefig(dirDict['classifyDir'] + titleStr + '.png', dpi=300, format='png', bbox_inches='tight')
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
    
    plt.savefig(dirDict['classifyDir'] + titleStr + '.png', dpi=300, format='png', bbox_inches='tight')
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
    
    plt.savefig(os.path.join(dirDict['outDir_model'], titleStr + '.svg'), dpi=300, format='svg', bbox_inches='tight')
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

        plt.savefig(dirDict['classifyDir'] + f"{titleStr}.{dirDict['outputFormat']}", dpi=300, format=dirDict['outputFormat'], bbox_inches='tight')
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

        plt.savefig(dirDict['classifyDir'] + f"{titleStr}.{dirDict['outputFormat']}", dpi=300, format=dirDict['outputFormat'], bbox_inches='tight')
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

    plt.savefig(dirDict['classifyDir'] + fullTitleStr + dirDict['outputFormat'], dpi=300,
                format=dirDict['outputFormat'], bbox_inches='tight')

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
    # plt.savefig(dirDict['classifyDir'] + fullTitleStr + '.png', dpi=300,
    #             format='png', bbox_inches='tight')

    # [left, bottom, width, height] of the colorbar axis
    cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig(dirDict['classifyDir'] + fullTitleStr + '.png', dpi=300, format='png', bbox_inches='tight')
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

    figSizeMat = np.array(mean_of_conf_matrix_arrays.shape)/2.2
    figSizeMat[0] = figSizeMat[0] + 1
    plt.figure(figsize=figSizeMat)
    ax = sns.heatmap(mean_of_conf_matrix_arrays, linewidth=0.25,cmap='Reds', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    ax.set(xticklabels=YtickLabs, yticklabels=YtickLabs, xlabel='Predicted Label', ylabel='True Label')
    plt.title(fullTitleStr, fontsize=figSizeMat[0]*1.5)

    # Save the plot
    plt.savefig(join(dirDict['outDir_model'], f"ConfusionMatrix_{fit}.{dirDict['outputFormat']}"), format=dirDict['outputFormat'], bbox_inches='tight')     
    plt.show()

def plotPRcurve(n_classes, y_real, y_prob, labelDict, daObjstr, fit, dirDict):
    # n_classes = int, number of classes
    # y_real, y_prob = test set labels and probabilities assigned to test set samples.
    # y_real, y_prob are in a [n_splits, n_samples, n_classes] format

    from sklearn.metrics import precision_recall_curve, auc
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.font_manager import FontProperties

    # Convert the arrays
    y_real = np.array(y_real)
    y_prob = np.array(y_prob)

    y_real_all, y_prob_all = [], []

    figSizeMat = np.array((n_classes, n_classes))/2.2
    f = plt.figure(figsize=figSizeMat)
    axes = plt.axes()

    # Depending on feature list y passed, determine if the classes are numbers or strings
    if all(isinstance(key, str) for key in labelDict.keys()):
        labelDict = {value: key for key, value in labelDict.items()}

    auc_dict = dict()

    colorDict = hf.create_color_dict(dictType='drug', rgbSwitch=0, alpha_value=0, scaleVal=False)

    for i in np.arange(n_classes):
        label_per_split = y_real[:, :, i]
        prob_per_split = y_prob[:, :, i]

        label_per_split_reshape = label_per_split.reshape(prob_per_split.shape[0]*prob_per_split.shape[1], 1)
        prob_per_split_reshape = prob_per_split.reshape(prob_per_split.shape[0]*prob_per_split.shape[1], 1)

        y_real_all.append(label_per_split_reshape)
        y_prob_all.append(prob_per_split_reshape)

        # Create PR curve
        precision, recall, _ = precision_recall_curve(label_per_split_reshape, prob_per_split_reshape)
        auc_val = auc(recall, precision)
        lab = f'{labelDict[i]}=%.2f' % (auc_val)
        auc_dict[labelDict[i]] = np.round(auc_val, 2)
        axes.step(recall, precision, label=lab, color=colorDict[labelDict[i]], lw=2)

    # Create a mean PR Curve by combining all the data
    precision, recall, _ = precision_recall_curve(np.concatenate(y_real_all), np.concatenate(y_prob_all))

    auc_val_mean = auc(recall, precision)
    lab = f'Mean Curve=%.2f' % (auc_val_mean)
    auc_dict['Mean'] = np.round(auc_val_mean, 2)
    axes.step(recall, precision, label=lab, lw=3, color='black')

    # PR Curves
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')

    if n_classes == 2:
        legend = axes.legend(loc='lower left', fontsize='xx-large')
        # Adjust the size for purposes of the paper
        for label in legend.get_texts():
            label.set_fontproperties(FontProperties(size=30, weight='bold'))
        for label in legend.get_lines():
            label.set_linewidth(15)
    else:
        legend = axes.legend(loc='lower left', fontsize='large')
        for label in legend.get_lines():
            label.set_linewidth(.5)

    plt.savefig(join(dirDict['outDir_model'], f"PRcurve_{fit}.{dirDict['outputFormat']}"), format=dirDict['outputFormat'], bbox_inches='tight')     
    plt.show()

    return auc_dict

def plot_shap_summary(X_train_trans_list, shap_values_list, y_vec, n_classes, plotDict, classifyDict, dirDict):
    
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

def plot_shap_force(X_train_trans_list, shap_values_list, baseline_val, y_real, numYDict, plotDict, dirDict):
    import itertools

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
    regionListSet = ['VISpm', 'LH', 'MA']

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

            y_idx = np.argmax(y_real[cvSplit], axis=1)
            y_labels = [labelDict[x] for x in y_idx]
            idStr = f"CV{cvSplit}_Sample{testPoint}"
            titleStr = f'Test Sample of {y_labels[testPoint]}, {idStr}'

            # Extract some key variables
            featCount = shap_values_list[cvSplit].shape[1]
            shapVals = np.round(shap_values_list[cvSplit].iloc[testPoint,1:featCount].values, 2)
            testVals = np.round(X_train_trans_list[cvSplit].iloc[testPoint,1:featCount].values, 2)
            featNames = list(shap_values_list[cvSplit].columns[1:featCount])

            # Plot
            shap.plots.force(baseline_val[cvSplit], shap_values=shapVals, features=testVals, feature_names=featNames, out_names=None, link='identity', plot_cmap='RdBu', matplotlib=True, show = False)

            # Modify the plot
            plt.title(titleStr, y=1.5, fontdict={'fontsize': 20})

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
    plt.savefig(join(dirDict['outDir_model'], f"featureCountHist.svg"), format='svg', bbox_inches='tight')
    plt.show()

def plot_cross_model_AUC(scoreNames, aucScores, aucScrambleScores, colorsList, saveDir):

    plt.figure(figsize=(5, 5))  # Adjust the width and height as needed
    plt.barh(scoreNames, aucScores, label='Data', color=colorsList[0])
    plt.barh(scoreNames, aucScrambleScores, label='Shuffled', color=colorsList[1])

    # Add legend
    plt.legend()

    # Set labels and title,
    plt.xlabel('Mean Area Under Precision-Recall Curve')
    plt.ylabel('Classifier')

    for index, value in enumerate(aucScores):
        percentage_text = '{:.0%}'.format(value)  # Format the value as a percentage
        plt.text(value-.01, index, percentage_text, ha='right', va='center', weight='bold', fontsize=10)

    for index, value in enumerate(aucScrambleScores):
        percentage_text = '{:.0%}'.format(value)  # Format the value as a percentage
        plt.text(value-.01, index, percentage_text, ha='right', va='center', weight='bold', fontsize=10)

    # Display the plot
    plt.savefig(f"{saveDir}MeanAUC_barplot.svg", format='svg', bbox_inches='tight')     
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
    plt.savefig(f"{saveDir}MeanAcc_barplot.svg", format='svg', bbox_inches='tight')     
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

    # Create a data frame with melted data
    flat_data = [item for sublist in data for item in sublist]

    df = pd.melt(pd.DataFrame(data, index=scoreNames).T, var_name='Category', value_name='Values')

    # Create horizontally oriented violin plot
    plt.figure(figsize=(5, 5))  # Adjust the width and height as needed

    ax = sns.violinplot(x='Values', y='Category', bw_adjust=.5, data=df, orient='h', color=colorsList[0])  #, palette=colors)  # Remove inner bars and set color
    # for violin in ax.collections:
    #     violin.set_alpha(1)

    # Set plot labels and title
    plt.xlabel('Number of Regions in Classifier')
    plt.ylabel('Classifier')

    plt.savefig(f"{dirDict['crossComp_figDir']}\\RegionCountPerSplit_violin.svg", format='svg', bbox_inches='tight')     

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
    plt.savefig(f"{dirDict['crossComp_figDir']}\\MeanSimilarity_heatmap.svg", format='svg', bbox_inches='tight')     
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
            plt.savefig(f"{dirDict['crossComp_figDir']}\\{figName}.svg", format='svg', bbox_inches='tight')     

            # Display the plot
            plt.show()

def plot_featureHeatMap(df_raw, scoreNames, featureLists, filterByFreq, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # Creates the heatmap for the data

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 9
    plt.rcParams['svg.fonttype'] = 'none'

    # Set variables
    dataFeature = 'abbreviation'
    blockCount = 2

    sys.path.append('../dependencies/')

    # Create a sorted structure from the data for scaffolding the desired heatmap
    brainAreaColorDict = hf.create_color_dict(dictType='brainArea', rgbSwitch=0)
    brainAreaPlotDict = hf.create_brainArea_dict('short')
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

    # Populate df_frame with 0s
    for col in df_frame.columns:
        df_frame[col] = 0
        
    for idx, (comp, featureList) in enumerate(zip(df_frame.columns, featureListDicts)):
        for regionName in featureList.keys():
            df_frame.loc[regionName, comp] = featureList[regionName]
        # df_frame[comp] = df_frame.index.isin(featureListArray[idx])
        # print(f"{comp}: {df_frame[comp].sum()}")
            
    # Remove any rows which are not above threshold
    df_plot = df_frame[df_frame.sum(axis=1) >= filterByFreq]
        
    # Remove the abbreviations from regionArea not represented in df_plot, Filter the regionArea for 'Cortex' and 'Thalamus'
    regionArea = hf.create_region_to_area_dict(df_raw, dataFeature)
    regionArea = regionArea[regionArea['abbreviation'].isin(df_plot.index)]
    regionArea = regionArea[regionArea['Brain_Area'].isin(['Cortex', 'Thalamus'])]

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]
    modelCount = len(df_plot.columns)

    # Create indicies for dividing the data into the correct number of sections regardless of the size
    row_idx_set = np.zeros((blockCount, 2), dtype=int)
    indices = np.linspace(0, len(df_plot), num=blockCount+1, dtype=int)
    for block_idx in range(blockCount):
        row_idx_set[block_idx][0] = indices[block_idx]
        row_idx_set[block_idx][1] = indices[block_idx+1]

    # Hand change to make Cortex 1st block, Thal 2nd
    row_idx_set[0,1] = 25
    row_idx_set[1,0] = 25

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
    df_plot = df_plot.drop(columns=['Brain_Area_Idx', 'Brain_Area'])

    # Drop the columns which do not include the string 'PSI'
    df_plot = df_plot.loc[:, df_plot.columns.str.contains('PSI')]

    # Update the column names to have 'PSI' in front.
    origcolNames = df_plot.columns
    colNames = [x.split(' vs ') for x in df_plot.columns]
    newColNames = [f'{x[1]} vs {x[0]}' for x in colNames]
    newColNames[-1], newColNames[-2] = origcolNames[-1], origcolNames[-2]
    newColNames = [x.replace('/', ' & ') for x in newColNames]
    df_plot.columns = newColNames

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    formatter.set_powerlimits((-2, 2))

    scalefactor = 12
    figH = (scalefactor*2.5)/blockCount
    figW = blockCount * 2.5

    colorbar = [False, True]

    for cbs in colorbar:

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

            heatmap = sns.heatmap(matrix, cmap='crest', ax=axes[idx] , fmt='.2f', cbar = False, square=True, yticklabels=yticklabels, xticklabels=xticklabels, cbar_kws={"format": formatter}, center=0)
            horzLineColor = 'black'

            # Rotate the xticklabels 45 degrees
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            # Add a colorbar
            # Change the colorbar labels to be in non-scientific notation

            if cbs:
                cbar = heatmap.figure.colorbar(heatmap.collections[0], ax=axes[idx], location='right', use_gridspec=True, pad=0.05)
                cbar.set_label('Feature Count', rotation=270, labelpad=5)
                cbar.ax.yaxis.set_major_formatter(formatter)

            # Add in horizontal lines breaking up brain regions types.
            line_break_num, line_break_ind = np.unique(region_idx, return_index=True)
            for l_idx in line_break_ind[1:]:
                axes[idx].axhline(y=l_idx, color=horzLineColor, linewidth=1)
                
            # Set the yl abel on the first subplot.
            # if idx == 0:
            #     axes[idx].set_ylabel("Region Names", fontsize=20)

            # if idx == 2:
            #     cbar = heatmap.collections[0].colorbar
            #     cbar.set_label('Colorbar Label', rotation=270, labelpad=5)

        titleStr = f"FeatureCountHeatmap"  
        # fig.suptitle(titleStr, fontsize=20, y=1)
        # fig.text(0.5, -.02, "Samples Per Group", ha='center', fontsize=20)
        plt.tight_layout(h_pad = 0, w_pad = .5)

        plt.savefig(f"{dirDict['crossComp_figDir']}{titleStr}_cb_{cbs}.svg", dpi=300, format='svg', bbox_inches='tight')
        plt.show()

def sortShap(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    min_value = float('inf')  # Initialize min_value to store the minimum value
    max_index = None  # Initialize max_index to store the corresponding index in shap_values_list
    min_index = None  # Initialize min_index to store the corresponding index in shap_values_list
    max_row_index = None  # Initialize max_row_index to store the index of the row with the maximum values
    min_row_index = None  # Initialize min_row_index to store the index of the row with the minimum values

    idxList = []

    normalizedVals = []
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
            min_index = idx
            min_row_index = row_sums.idxmin()

    max_index = idxList[-1]

    return databaseIdx