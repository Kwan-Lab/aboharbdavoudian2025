import pandas as pd
import numpy as np
from os.path import exists
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import helperFunctions as hf
import scipy.stats as stats

def totalCountsPlot(pandasdf, column2Plot, dirDict, outputFormat):
    # first calculate total cells for drug/sex/etc # Needs fixing, this group is gone. Use a sum or something.
    totalCellCountData = pandasdf.groupby(['dataset', 'drug', 'sex'])[
        column2Plot].mean().reset_index()

    drugList = list(pandasdf.drug.unique())

    # plotting
    sns.set(font_scale=2)
    sns.set_style('ticks')
    sns.despine()

    markers = {'f': 'v', 'm': '^'}
    plt.figure(figsize=(15, 15))
    order = drugList

    # if logSwitch == True:
    #     totalCellCountData['log_total_cells'] = np.log(totalCellCountData['total_cells'])
    #     yVar = 'log_total_cells'
    # else:
    yVar = 'total_cells'

    ax = sns.boxplot(x="drug", y=yVar, data=totalCellCountData, whis=0, dodge=False, showfliers=False, linewidth=0.5,
                     hue='drug', palette=sns.color_palette(n_colors=len(drugList)))
    sns.scatterplot(x="drug", y=yVar, data=totalCellCountData, hue='drug', linewidth=0, style='sex', markers=True,
                    s=300, palette=sns.color_palette(n_colors=len(drugList)), ax=ax)
    # sns.stripplot(x="drug", y="total_cells", data=lightsheet_whole_brain)
    # ax = sns.swarmplot(x="drug", y="total_cells", data=lightsheet_whole_brain, hue='drug', linewidth=0, palette=sns.color_palette(n_colors=len(drugList)), zorder=.5)

    # lower opacity
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))

    # remove legend
    plt.legend([], [], frameon=False)

    # cleanup
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.xaxis.set_tick_params(length=20, width=0.5)
    ax.yaxis.set_tick_params(length=20, width=0.5)
    ax.set_xlabel('Drug')
    ax.set_ylabel('Total Cells (Count)')
    # if logSwitch == True:
    #     ax.set_ylabel('log(total cells)')
    # else:
    ax.set_yscale('log')

    ax.set(ylim=(5e5, 1e7))
    sns.despine()

    plt.savefig(dirDict['outDir'] + 'totalCells.' +
                outputFormat,  format=outputFormat, bbox_inches='tight')

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
    brainAreaListPlot= ['Olfactory', 'Cortex', 'Hippo', 'Stri+Pall', 'Thalamus', 'Hypothalamus', 'Mid Hind Medulla', 'Cerebellum']
    brainAreaColor =     ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c'] #, '#dede00'
    brainAreaPlotDict = dict(zip(brainAreaList, brainAreaListPlot))
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

def data_heatmap(lightsheet_data, classifyDict, dirDict):

    # Create a list of unique 'Region_Name' and 'Brain_Area' values
    unique_brain_areas = lightsheet_data['Brain_Area'].unique()

    for brain_area in unique_brain_areas:
        df = lightsheet_data.loc[lightsheet_data['Brain_Area'] == brain_area]
        df_Tilted = df.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])

        # Test
        matrix = df_Tilted.values.T

        yticklabels = df_Tilted.columns.values.tolist()
        xticklabels = df_Tilted.index.values.tolist()
        xticklabels = [x[0:-1] for x in xticklabels]

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
        
        titleStr = f"{classifyDict['data']} in {brain_area}"

        plt.ylabel("Feature Names (Region Names)")
        plt.xlabel("Samples Per Group", fontsize=12)
        plt.tick_params(axis='x', which='both', length=0)
        plt.title(titleStr, fontsize=15)
        
        # plt.savefig(dirDict['classifyDir'] + titleStr + '.png', dpi=300, format='png', bbox_inches='tight')
        plt.show()

def data_heatmap_single(lightsheet_data, classifyDict, dirDict):

    # Pivot data to represent samples, features, and data correctly for a heatmap.
    df_Tilted = lightsheet_data.pivot(index=classifyDict['feature'], columns='dataset', values=classifyDict['data'])

    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, classifyDict)

    # Sort accordingly
    df_Tilted = df_Tilted.loc[regionArea[classifyDict['feature']]]
    region_idx = regionArea.Brain_Area_Idx  # Extract for horizontal lines in plot later.

    # Extract data
    matrix = df_Tilted.values

    xticklabels = df_Tilted.columns.values.tolist()
    xticklabels = [x[0:-1] for x in xticklabels]
    yticklabels = df_Tilted.index.values.tolist()

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
    
    # cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    sns.heatmap(matrix, cmap=cmap, fmt='.2f', yticklabels=yticklabels, xticklabels=x_labels)

    # Add in vertical lines breaking up sample types
    _, line_break_ind = np.unique(xticklabels, return_index=True)
    for idx in line_break_ind:
        plt.axvline(x=idx, color='white', linewidth=2)
    
    titleStr = f"{classifyDict['data']}"

    # Add in horizontal lines breaking up brain regions types.
    _, line_break_ind = np.unique(region_idx, return_index=True)
    for idx in line_break_ind:
        plt.axhline(y=idx, color='white', linewidth=2)
    
    titleStr = f"{classifyDict['data']}"
    plt.ylabel("Feature Names (Region Names)")
    plt.xlabel("Samples Per Group", fontsize=12)
    plt.tick_params(axis='x', which='both', length=0)
    plt.title(titleStr, fontsize=15)
    
    # plt.savefig(dirDict['classifyDir'] + titleStr + '.png', dpi=300, format='png', bbox_inches='tight')
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

    plt.savefig(dirDict['classifyDir'] + fullTitleStr + '.png', dpi=300,
                format='png', bbox_inches='tight')

    # Add a colorbar on the far right plot
    vmin = np.min(matrix)
    vmax = np.max(matrix)

    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])

    # Not working
    # plt.savefig(dirDict['classifyDir'] + fullTitleStr + '.png', dpi=300,
    #             format='png', bbox_inches='tight')

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

    figSizeMat = np.array(mean_of_conf_matrix_arrays.shape)+1
    plt.figure(figsize=figSizeMat)
    ax = sns.heatmap(mean_of_conf_matrix_arrays, linewidth=0.25,cmap='coolwarm', annot=True, fmt=".2f")
    ax.set(xticklabels=YtickLabs, yticklabels=YtickLabs, xlabel='Predicted Label', ylabel='True Label')
    plt.title(fullTitleStr, fontsize=sum(figSizeMat))

    # Save the plot
    plt.savefig(dirDict['classifyDir'] + titleStr + '.png',format='png', bbox_inches='tight')     
    plt.show()

def plotPRcurve(n_classes, y_real, y_prob, labelDict, daObjstr):
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

    for i in np.arange(n_classes):
        label_per_split = y_real[:, :, i]
        prob_per_split = y_prob[:, :, i]

        label_per_split_reshape = label_per_split.reshape(prob_per_split.shape[0]*prob_per_split.shape[1], 1)
        prob_per_split_reshape = prob_per_split.reshape(prob_per_split.shape[0]*prob_per_split.shape[1], 1)

        y_real_all.append(label_per_split_reshape)
        y_prob_all.append(prob_per_split_reshape)

        # Create PR curve
        precision, recall, _ = precision_recall_curve(label_per_split_reshape, prob_per_split_reshape)
        lab = f'{labelDict[i]} AUC=%.2f' % (auc(recall, precision))
        axes.step(recall, precision, label=lab, lw=2)

    # Create PR curve
    precision, recall, _ = precision_recall_curve(np.concatenate(y_real_all), np.concatenate(y_prob_all))

    lab = f'Mean Curve AUC=%.2f' % (auc(recall, precision))
    axes.step(recall, precision, label=lab, lw=3, color='black')

    # PR Curves
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.legend(loc='lower left', fontsize='small')
    axes.set_title(daObjstr + ', PR Curves')

    plt.show()
