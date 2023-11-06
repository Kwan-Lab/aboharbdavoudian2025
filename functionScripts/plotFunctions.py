import pandas as pd
import numpy as np
from os.path import exists, join
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.ticker as tkr
import helperFunctions as hf
import scipy.stats as stats
import shap
import os
import matplotlib.gridspec as gridspec

def totalCountsPlot(pandasdf, column2Plot, dirDict, outputFormat):
    # first calculate total cells for drug/sex/etc # Needs fixing, this group is gone. Use a sum or something.
    # totalCellCountData = pandasdf.groupby(['dataset', 'drug', 'sex'])[column2Plot].mean().reset_index()
    totalCellCountData = pandasdf[pandasdf.Region_ID == 88]

    drugList = list(pandasdf.drug.unique())

    # plotting
    sns.set(font_scale=2)
    sns.set_style('ticks')
    sns.despine()

    markers = {'f': 'v', 'm': '^'}
    plt.figure(figsize=(10, 3.5))
    order = drugList

    # if logSwitch == True:
    #     totalCellCountData['log_total_cells'] = np.log(totalCellCountData['total_cells'])
    #     yVar = 'log_total_cells'
    # else:
    yVar = 'total_cells'
    colorPal = sns.color_palette(n_colors=len(drugList))

    ax = sns.boxplot(x="drug", y=yVar, data=totalCellCountData, whis=0, dodge=False, showfliers=False, linewidth=1, saturation=1, hue='drug', palette=colorPal)
    sns.scatterplot(x="drug", y=yVar, data=totalCellCountData, hue='drug', linewidth=1, style='sex', markers=True, s=100, palette=colorPal, ax=ax, edgecolor='black')

    # sns.stripplot(x="drug", y="total_cells", data=lightsheet_whole_brain)
    # ax = sns.swarmplot(x="drug", y="total_cells", data=lightsheet_whole_brain, hue='drug', linewidth=0, palette=sns.color_palette(n_colors=len(drugList)), zorder=.5)

    # lower opacity
    # for patch in ax.artists:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_alpha(1)
    #     # patch.set_facecolor((r, g, b, 1))

    # remove legend
    plt.legend([], [], frameon=False)

    # cleanup
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.xaxis.set_tick_params(length=20, width=0.5)
    ax.yaxis.set_tick_params(length=20, width=0.5)
    ax.set_xlabel('Drug', fontdict={'fontsize':20})
    ax.set_ylabel('Total Cells (Count)', fontdict={'fontsize':20})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)

    # if logSwitch == True:
    #     ax.set_ylabel('log(total cells)')
    # else:
    ax.set_yscale('log')

    ax.set(ylim=(5e5, 1e7))
    sns.despine()

    plt.savefig(dirDict['outDir'] + 'totalCells.' + outputFormat,  format=outputFormat, bbox_inches='tight')

    sns.set_theme()

def plotLowDimEmbed(pandasdf, column2Plot, dirDict, dimRedMeth, classifyDict):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    from itertools import combinations

    # If some filtering is desired, do so here
    if classifyDict['featurefilt']:
        # vmax = np.percentile(pandasdf[column2Plot], 99.9)
        # pandasdf_over = pandasdf[pandasdf[column2Plot] > vmax]
        # features_over = pandasdf_over['abbreviation'].unique()
        # pandasdf = pandasdf[~pandasdf['abbreviation'].isin(features_over)]
        pandasdf = hf.filter_features(pandasdf, classifyDict)
        filtTag = 'filt'
    else:
        filtTag = ''

    
    # Pivot the lightsheet data table
    df_Tilted = pandasdf.pivot(index='dataset', columns='abbreviation', values=column2Plot)
    n_comp = 4

    # Perform dimensionality reduction
    # Scale features beforehand
    X_scaled = RobustScaler().fit_transform(df_Tilted)
    # X_scaled_pd = pd.DataFrame(X_scaled, index=df_Tilted.index, columns=df_Tilted.columns)
    # sns.pairplot(X_scaled_pd)
    # plt.show()

    y = np.array([x[:-1] for x in df_Tilted.index])
    y = ['5MEO' if item == 'DMT' else item for item in y]
    y = ['6-F-DET' if item == '6FDET' else item for item in y]

    # Visualize the scaled features
    if dimRedMeth == 'PCA':
        pca = PCA(n_components=n_comp)
        X_scaled_dimRed = pca.fit_transform(X_scaled)
    elif dimRedMeth == 'LDA':
        lda = LDA(n_components=n_comp)
        X_scaled_dimRed = lda.fit_transform(X_scaled, y)
    else:
        KeyError('dimRedMethod not recognized, pick LDA or PCA')

    # Create average cases for each drug.
    compName = dimRedMeth[0:2]
    colNames = [f"{compName}{x}" for x in range(1, n_comp+1)]

    dimRedData = pd.DataFrame(data=X_scaled_dimRed, index=df_Tilted.index, columns=colNames)
    dimRedData.loc[:, 'drug'] = y
    dimRedDrugMean = dimRedData.groupby(by='drug').mean()

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
        sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData, s=50, alpha=0.75)
        sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean, s=100, legend=False, edgecolor='black')
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
    brainAreaList= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']
    brainAreaColor =     ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c'] #, '#dede00'
    brainAreaPlotDict = hf.create_brainArea_dict('short')
    
    brainAreaColorDict = dict(zip(brainAreaList, brainAreaColor))

    regionArea = hf.create_region_to_area_dict(lightsheet_data, classifyDict)
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

    # Create an index for sorting the brain Areas. List below is for custom ordering.
    # brainAreaList = lightsheet_data['Brain_Area'].unique().tolist()
    brainAreaList= ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum', 'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum']
    brainAreaListPlot= ['Olfactory', 'Cortex', 'Hippo', 'Stri+Pall', 'Thalamus', 'Hypothalamus', 'Mid Hind Medulla', 'Cerebellum']
    brainAreaColor =     ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c'] #, '#dede00'
    brainAreaPlotDict = dict(zip(brainAreaList, brainAreaListPlot))
    brainAreaColorDict = dict(zip(brainAreaList, brainAreaColor))
    AreaIdx = dict(zip(brainAreaList, np.arange(len(brainAreaList))))

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
    
    # plt.savefig(dirDict['classifyDir'] + titleStr + '.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()

def data_heatmap(lightsheet_data, dataFeature, dataValues, dirDict):

    # Create a list of unique 'Region_Name' and 'Brain_Area' values
    unique_brain_areas = lightsheet_data['Brain_Area'].unique()

    for brain_area in unique_brain_areas:
        df = lightsheet_data.loc[lightsheet_data['Brain_Area'] == brain_area]
        df_Tilted = df.pivot(index='dataset', columns=dataFeature, values=dataValues)

        # Test
        matrix = df_Tilted.values.T

        yticklabels = df_Tilted.columns.values.tolist()
        xticklabels = df_Tilted.index.values.tolist()
        xticklabels = [x[0:-1] for x in xticklabels]

        # Correct DMT Label to 5-MeO DMT
        xticklabels = ['5-MeO-DMT' if item == 'DMT' else item for item in xticklabels]
        xticklabels = ['6-F-DET' if item == '6FDET' else item for item in xticklabels]

        # Convert to x axis labels
        x_labels = ['' for _ in range(matrix.shape[1])]
        result = hf.find_middle_occurrences(xticklabels)
        for mid_sample_ind in result:
            x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
            
        # Plotting variables
        scalefactor = 12
        vmin, vmax = np.percentile(matrix.flatten(), [5, 99])
        cmap = 'rocket'

        # Plotting
        plt.figure(figsize=(scalefactor, len(yticklabels) * scalefactor * 0.0125))
        
        # cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        sns.heatmap(matrix, cmap=cmap, vmin=vmin, vmax=vmax, fmt='.2f', yticklabels=yticklabels, xticklabels=x_labels, square=True)

        # Add in vertical lines breaking up sample types
        _, line_break_ind = np.unique(xticklabels, return_index=True)
        for idx in line_break_ind:
            plt.axvline(x=idx, color='white', linewidth=2)
        
        titleStr = f"{dataValues} in {brain_area}"

        plt.ylabel("Feature Names (Region Names)")
        plt.xlabel("Samples Per Group", fontsize=12)
        plt.tick_params(axis='x', which='both', length=0)
        plt.title(titleStr, fontsize=15)
        
        plt.savefig(dirDict['classifyDir'] + titleStr + '.svg', dpi=300, format='svg', bbox_inches='tight')
        plt.show()

def data_heatmap_single(lightsheet_data, dataFeature, dataValues, dirDict):

    colorMapCap = False

    # Pivot data to represent samples, features, and data correctly for a heatmap.
    df_Tilted = lightsheet_data.pivot(index=dataFeature, columns='dataset', values=dataValues)

    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, dataFeature)

    # Sort accordingly
    df_Tilted = df_Tilted.loc[regionArea[dataFeature]]
    region_idx = regionArea.Brain_Area_Idx  # Extract for horizontal lines in plot later.

    # Extract data
    matrix = df_Tilted.values

    xticklabels = df_Tilted.columns.values.tolist()
    yticklabels = df_Tilted.index.values.tolist()

    xticklabels = [x[0:-1] for x in xticklabels]

    # Correct DMT Label to 5-MeO DMT
    xticklabels = ['5-MeO-DMT' if item == 'DMT' else item for item in xticklabels]
    xticklabels = ['6-F-DET' if item == '6FDET' else item for item in xticklabels]

    # Convert to x axis labels
    x_labels = ['' for _ in range(matrix.shape[1])]
    result = hf.find_middle_occurrences(xticklabels)
    for mid_sample_ind in result:
        x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
        
    # Plotting variables
    scalefactor = 12
    cmap = 'rocket'

    # Plotting
    plt.figure(figsize=(scalefactor, len(yticklabels) * scalefactor * 0.0125))

    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    if colorMapCap:
        vmin, vmax = np.percentile(matrix.flatten(), [5, 99])
        sns.heatmap(matrix, cmap=cmap, fmt='.2f', yticklabels=yticklabels, xticklabels=x_labels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter})
    else:
        sns.heatmap(matrix, cmap=cmap, fmt='.2f', yticklabels=yticklabels, xticklabels=x_labels, cbar_kws={"format": formatter})

    # Add in vertical lines breaking up sample types
    _, line_break_ind = np.unique(xticklabels, return_index=True)
    for idx in line_break_ind:
        plt.axvline(x=idx, color='white', linewidth=2)
    
    titleStr = f"{dataValues}"

    # Add in horizontal lines breaking up brain regions types.
    _, line_break_ind = np.unique(region_idx, return_index=True)
    for idx in line_break_ind:
        plt.axhline(y=idx, color='white', linewidth=2)
    
    titleStr = f"{dataValues}"
    plt.ylabel("Feature Names (Region Names)")
    plt.xlabel("Samples Per Group", fontsize=12)
    plt.tick_params(axis='x', which='both', length=0)
    plt.title(titleStr, fontsize=15)
    
    plt.savefig(dirDict['classifyDir'] + titleStr + '.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.show()

def data_heatmap_block(lightsheet_data, dataFeature, dataValues, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # TODO: shift code to use 'GridSpec' and create a single image with 3 equally sized columns and a colorbar at once. 
    # Creates the heatmap for the data, arranged to be 3 columns. 
    colorMapCap = True

    # Pivot data to represent samples, features, and data correctly for a heatmap.
    df_Tilted_all = lightsheet_data.pivot(index=dataFeature, columns='dataset', values=dataValues)

    row_idx_set = [[0, 105], [106, 210], [211, 315]]

    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, dataFeature)

    # Sort the data to be combined per larger area
    df_Tilted_all = df_Tilted_all.loc[regionArea[dataFeature]]

    # Resort for coherence across figures
    newList =['PSI', 'KET', 'DMT', '6FDET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']
    newListnum = [f'{item}{i}' for item in newList for i in range(1, 8 + 1)]
    df_Tilted_all = df_Tilted_all[newListnum]

    # Find the ends of the colormap
    if colorMapCap:
        vmin, vmax = np.percentile(df_Tilted_all.values.flatten(), [1, 99])
    else:
        vmin = df_Tilted_all.min().min()
        vmax = df_Tilted_all.max().max()


    scalefactor = 12
    # Plotting variables
    cmap = 'rocket'
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # Create a version with and without the colorbar
    colorBarSwitch = [True, False]

    for cbs in colorBarSwitch:

        fig, axes = plt.subplots(1, 3, figsize=(scalefactor*2.4, len(df_Tilted_all)/len(row_idx_set) * scalefactor * 0.0125))  # Adjust figsize as needed

        for idx, row_set in enumerate(row_idx_set):

            # Slice and modify previous structures to create segment
            df_Tilted = df_Tilted_all.iloc[row_set[0]: row_set[1], :]
            regionArea_local = regionArea[regionArea[dataFeature].isin(df_Tilted.index)]
            region_idx = regionArea_local.Brain_Area_Idx  # Extract for horizontal lines in plot later.

            # Extract data
            matrix = df_Tilted.values

            xticklabels = df_Tilted.columns.values.tolist()
            yticklabels = df_Tilted.index.values.tolist()

            xticklabels = [x[0:-1] for x in xticklabels]

            # Correct DMT Label to 5-MeO DMT
            xticklabels = ['5MEO' if item == 'DMT' else item for item in xticklabels]
            xticklabels = ['6-F-DET' if item == '6FDET' else item for item in xticklabels]

            # Convert to x axis labels
            x_labels = ['' for _ in range(matrix.shape[1])]
            result = hf.find_middle_occurrences(xticklabels)
            for mid_sample_ind in result:
                x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
                
            heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx] ,fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=x_labels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter})

            # Clear the x-ticks
            heatmap.tick_params(axis='x', which='both', length=0, labelsize=14)

            # Add in vertical lines breaking up sample types
            _, line_break_ind = np.unique(xticklabels, return_index=True)
            for l_idx in line_break_ind:
                axes[idx].axvline(x=l_idx, color='white', linewidth=1)
            
            # Add in horizontal lines breaking up brain regions types.
            _, line_break_ind = np.unique(region_idx, return_index=True)
            for l_idx in line_break_ind:
                axes[idx].axhline(y=l_idx, color='white', linewidth=1)
            
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

    figSizeMat = np.array(mean_of_conf_matrix_arrays.shape)
    figSizeMat[0] = figSizeMat[0] + 1
    plt.figure(figsize=figSizeMat)
    ax = sns.heatmap(mean_of_conf_matrix_arrays, linewidth=0.25,cmap='coolwarm', annot=True, fmt=".2f", square=True)
    ax.set(xticklabels=YtickLabs, yticklabels=YtickLabs, xlabel='Predicted Label', ylabel='True Label')
    plt.title(fullTitleStr, fontsize=figSizeMat[0]*1.5)

    # Save the plot
    plt.savefig(join(dirDict['outDir_model'], f"{titleStr}.{dirDict['outputFormat']}"), format=dirDict['outputFormat'], bbox_inches='tight')     
    plt.show()

def plotPRcurve(n_classes, y_real, y_prob, labelDict, daObjstr, dirDict):
    # n_classes = int, number of classes
    # y_real, y_prob = test set labels and probabilities assigned to test set samples.
    # y_real, y_prob are in a [n_splits, n_samples, n_classes] format

    from sklearn.metrics import precision_recall_curve, auc

    # Convert the arrays
    y_real = np.array(y_real)
    y_prob = np.array(y_prob)

    y_real_all = []
    y_prob_all = []

    f = plt.figure(figsize=(8, 8))
    axes = plt.axes()

    # Depending on feature list y passed, determine if the classes are numbers or strings
    if all(isinstance(key, str) for key in labelDict.keys()):
        labelDict = {value: key for key, value in labelDict.items()}

    auc_dict = dict()

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
        axes.step(recall, precision, label=lab, lw=2)

    # Create PR curve
    precision, recall, _ = precision_recall_curve(np.concatenate(y_real_all), np.concatenate(y_prob_all))

    auc_val_mean = auc(recall, precision)
    lab = f'Mean Curve=%.2f' % (auc_val_mean)
    auc_dict['Mean'] = np.round(auc_val_mean, 2)
    axes.step(recall, precision, label=lab, lw=3, color='black')

    # PR Curves
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.legend(loc='lower left', fontsize='small')
    axes.set_title(daObjstr + ', PR Curves')
    plt.savefig(join(dirDict['outDir_model'], f"PRcurve.{dirDict['outputFormat']}"), format=dirDict['outputFormat'], bbox_inches='tight')     

    plt.show()

    return auc_dict

def plotSHAPSummary(X_train_trans_list, shap_values_list, baseline_val, y_real, numYDict, n_classes, n_splits, dirDict):

    X_train_trans_nonmean = pd.concat(X_train_trans_list, axis=0)
    shap_values_nonmean = []
    if n_classes == 2:
        shap_values_nonmean.append(pd.concat(shap_values_list, axis=0))
        test_count = shap_values_list[0].shape[0]
    else:
        test_count = shap_values_list[0][0].shape[0]
        for shap_x_df in shap_values_list:
            shap_values_nonmean.append(pd.concat(shap_x_df, axis=0))

    # Do an example force plot
    cvSplit = 2

    # Generate an array of true labels
    labelDict = {value: key for key, value in numYDict.items()}
    y_idx = np.argmax(y_real[cvSplit], axis=1)
    y_labels = [labelDict[x] for x in y_idx]

    if n_classes == 2:
        for testPoint in range(0,4):
            featCount = shap_values_list[cvSplit].shape[1]
            shapVals = np.round(shap_values_list[cvSplit].iloc[testPoint,1:featCount].values, 2)
            testVals = np.round(X_train_trans_list[cvSplit].iloc[testPoint,1:featCount].values, 2)
            featNames = list(shap_values_list[cvSplit].columns[1:featCount])
            shap.plots.force(baseline_val[cvSplit], shap_values=shapVals, features=testVals, feature_names=featNames, out_names=None, link='identity', plot_cmap='RdBu', matplotlib=True, show = False)
            plt.title(f'Test Sample of {y_labels[testPoint]}', y=1.5, fontdict={'fontsize': 20})
            plt.show()

    # Plot the SHAP values for each class
    for shap_vals in shap_values_nonmean:
        # determine how many models across all the splits each feature was included in
        shapValueCount = shap_vals.agg(np.isnan).sum()
        feature_model_count = n_splits - shapValueCount/test_count
        shapValFeature_sorted = feature_model_count.sort_values(ascending=False)

        sortingIdx = shapValFeature_sorted.index[1:]
        testCaseCount = [int(x) for x in shapValFeature_sorted.values[1:]]
    
        X_train_trans_sorted = X_train_trans_nonmean.loc[:,sortingIdx]
        shap_values_sorted = shap_vals.loc[:,sortingIdx]

        # Adjust the feature names to include their counts.
        featureNames = [f"{feat} ({testCaseCount[idx]})" for idx, feat in enumerate(sortingIdx)]
        shap.summary_plot(shap_values_sorted.values, X_train_trans_sorted.values, feature_names=featureNames, sort=False, show=False)
        # plt.title('SHAP Values, Test data', fontdict={'fontsize': 20})
        plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary.svg"), format='svg', bbox_inches='tight')
        plt.show()

def plot_feature_scores(clf, featureNames):
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