import os
import sys
import pandas as pd
import numpy as np
from os.path import exists
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def compareAnimals(lightsheetDB, drugA, drugB, compColumn, dirDict):
    # code which will create a grid of plots comparing individual animals.

    plotTitle = drugA + ' vs ' + drugB + ' - ' + compColumn

    drugAdata = lightsheetDB.loc[lightsheetDB.drug.str.match(drugA)]
    drugBdata = lightsheetDB.loc[lightsheetDB.drug.str.match(drugB)]

    # Identify datasets to iterate across.
    drugAsets = drugAdata.dataset.unique()
    drugBsets = drugBdata.dataset.unique()

    # Seperate the lightsheetDB based on individual animals
    fig, axs = plt.subplots(len(drugAsets), len(
        drugBsets), figsize=(8, 8), dpi=200)

    # drugAllsets = list(drugAsets) + list(drugBsets)
    # fig, axs = plt.subplots(len(drugAllsets), len(drugAllsets), figsize=(10, 10), dpi=200)
    fig.suptitle(plotTitle, y=0.92)

    fontdict2 = {'fontsize': 5}

    # Iterate across drugsets, creating correlations.
    # for drugA_i, drugA_set in enumerate(drugAllsets):
    #     for drugB_i, drugB_set in enumerate(drugAllsets):
    for drugA_i, drugA_set in enumerate(drugAsets):
        for drugB_i, drugB_set in enumerate(drugBsets):

            # Merge the datasets into a single table.
            drugAsetData = lightsheetDB.loc[lightsheetDB.dataset == drugA_set]
            drugBsetData = lightsheetDB.loc[lightsheetDB.dataset == drugB_set]

            # Convention to allow for self comparison
            colNameA = drugA_set + '_' + list(drugAsetData.sex)[0] + '_'
            colNameB = drugB_set + '_' + list(drugBsetData.sex)[0]

            # Rename the scaled cell density columns to the dataset.
            drugAsetData = drugAsetData.rename(columns={compColumn: colNameA})
            drugAsetData[colNameB] = drugBsetData[compColumn].to_numpy()

            # Identify regions with infs/nans/etc
            regions2Remove = drugAsetData['Region_Name'][drugAsetData.isin(
                [np.nan, np.inf, -np.inf]).any(1)]

            # Remove these regions from all output structures
            drugAsetData = drugAsetData[~drugAsetData['Region_Name'].isin(
                regions2Remove)]

            prctileCutOffX = np.percentile(
                drugAsetData[colNameA].to_numpy(), 95)
            prctileCutOffY = np.percentile(
                drugAsetData[colNameB].to_numpy(), 95)

            # Plot into the subaxes
            ax = sns.scatterplot(
                x=colNameA, y=colNameB, data=drugAsetData, s=5, ax=axs[drugB_i][drugA_i])

            # Add in Correlation coefficient
            tmpSet = np.vstack(
                (np.asarray(drugAsetData[colNameA]), np.asarray(drugAsetData[colNameB])))
            corrCoeffPlot = np.corrcoef(tmpSet)[0][1]

            ax.set_title('r = ' + str(round(corrCoeffPlot, 2)),
                         fontdict=fontdict2,  y=0.98)

            ax.tick_params(axis='x', labelsize=4)
            ax.tick_params(axis='y', labelsize=4)
            ax.set_xlim([0, prctileCutOffX])
            ax.set_ylim([0, prctileCutOffY])
            maxNum = np.amax([prctileCutOffX, prctileCutOffY])
            ax.plot([0, maxNum], [0, maxNum], linewidth=1, alpha=0.5)

            if not drugA_i == 0:
                ax.set(ylabel=None)

            if not drugB_i == len(drugB_set)-1:
                # if not drugB_i == len(drugAllsets)-1:
                ax.set(xlabel=None)

    saveTitle = plotTitle.replace("/", ' per ')
    plt.savefig(dirDict['debugDir'] + saveTitle + '.png',
                format='png', bbox_inches='tight')
    plt.show()


def drug_stats_and_changes(databaseFrame, drugList):
    # Function for detecting changes in activation density across 2 drug conditions.
    # Returns a table containing a column representing 'cell_density_change' across conditions per region((X - Y)/Y * 100).

    # Identify drugs in the dataset
    drugListInds = range(0, len(drugList))
    drugDataFrames = np.empty(len(drugList), dtype=object)
    drugStatsAll = np.empty(len(drugList), dtype=object)

    # Gather Statistics for them individually
    for drug_i in range(0, len(drugList)):
        drug = drugList[drug_i]
        drugTemp = databaseFrame[databaseFrame.drug == drug]
        drugStats = drugTemp.groupby(['Region_Name']).agg({'cell_density': ['mean', 'std', 'count']})
        drugStats.columns = drugStats.columns.droplevel(0)
        drugStats.columns = [drug + '_average', drug + '_standard_deviation', drug + '_observations']

        # Merge into single table
        if drug_i == 0:
            drugStatsAll = drugStats
        else:
            drugStatsAll = pd.merge(drugStatsAll, drugStats, on='Region_Name', how='inner')

        # Store for comparison later
        drugDataFrames[drug_i] = drugTemp

    drugStatsAll = drugStatsAll.reset_index()

    # Perform All possible comparisons
    drugPairList = [(a, b) for idx, a in enumerate(drugList) for b in drugList[idx + 1:]]
    drugPairIndList = [(a, b) for idx, a in enumerate(drugListInds) for b in drugListInds[idx + 1:]]
    drugPairDataList = np.empty(len(drugPairList), dtype=object)
    drugPairDataNames = np.empty(len(drugPairList), dtype=object)

    for comp_i in range(0, len(drugPairList)):
        # Merge the two tables.
        drugX_drugY = pd.merge(drugDataFrames[drugPairIndList[comp_i][0]],
                               drugDataFrames[drugPairIndList[comp_i][1]], on='Region_Name', how='inner')

        tmpDB = drugDataFrames[drugPairIndList[comp_i][0]]
        tmpDBFilt = tmpDB.loc[tmpDB['Region_Name'] == 'Dorsal nucleus raphe']

        tmpDB2 = drugDataFrames[drugPairIndList[comp_i][1]]
        tmpDB2Filt = tmpDB2.loc[tmpDB2['Region_Name']
                                == 'Dorsal nucleus raphe']

        tmpDB3 = drugX_drugY.loc[drugX_drugY['Region_Name']
                                 == 'Dorsal nucleus raphe']

        # Determine comparison name
        drugCompName = drugPairList[comp_i][0] + '-' + drugPairList[comp_i][1]

        # Calculate cell_density_change
        drugX_drugY[drugCompName] = ((drugX_drugY['cell_density_x'] -
                                     drugX_drugY['cell_density_y']) / drugX_drugY['cell_density_y']) * 100

        # Clean up the table
        drugX_drugY = drugX_drugY.drop(
            ['dataset_x', 'drug_x', 'cell_density_x', 'dataset_y', 'drug_y', 'cell_density_y'], axis=1)
        drugX_drugY = drugX_drugY.rename(
            columns={'abbreviation_x': 'abbreviation', 'Region_ID_x': 'Region_ID'})

        # select columns of interest
        column_of_interest = ['Region_ID',
                              'abbreviation', 'Region_Name', drugCompName]
        drugX_drugY = drugX_drugY[column_of_interest]

        # Average across regions for each drug comparison.
        drugX_drugY_meanPerRegion = drugX_drugY.groupby(
            ['Region_ID', 'abbreviation', 'Region_Name']).mean().reset_index()
        # drugX_drugY_meanPerRegion = drugX_drugY # Or dont!

        # Store in array, mainly to calculate error bars on differences in Figure 2. Putting out single Dataframe overwhelms memory (32 GB at least)
        # Additional Note: lists can't be merged due to different numbers in each drug group.
        drugPairDataList[comp_i] = drugX_drugY
        drugPairDataNames[comp_i] = drugCompName

        # Combine with data up to this point
        if comp_i == 0:
            drugPairDB = drugX_drugY_meanPerRegion
        else:
            drugPairDB = pd.merge(
                drugPairDB, drugX_drugY_meanPerRegion, on='Region_Name', how='inner')
            drugPairDB = drugPairDB.drop(
                ['Region_ID_y', 'abbreviation_y'], axis=1)
            drugPairDB = drugPairDB.rename(
                columns={'abbreviation_x': 'abbreviation', 'Region_ID_x': 'Region_ID'})

    # Clean up final table as needed
    # #remove duplicate column names
    drugPairDB = drugPairDB.loc[:, ~drugPairDB.columns.duplicated()]

    # Identify regions with infs/nans/etc
    regions2Remove = drugPairDB.Region_Name[drugPairDB.isin(
        [np.nan, np.inf, -np.inf]).any(1)]

    # Remove these regions from all output structures
    drugPairDB = drugPairDB[~drugPairDB['Region_Name'].isin(regions2Remove)]
    drugStatsAll = drugStatsAll[~drugStatsAll['Region_Name'].isin(
        regions2Remove)]
    for drugComp_i, drugComp in enumerate(drugPairDataList):
        drugPairDataList[drugComp_i] = drugComp[~drugComp['Region_Name'].isin(
            regions2Remove)]

    return drugPairDB, drugStatsAll, drugPairDataList, drugPairDataNames


def brainRegionTrend(lightsheetDBAll, dataColumn, drugA, drugB, ylimMax):
    # A function which generates bar plots for each 'clusteColumn' where data comes from 'dataColumn'.
    # dataColumn = 'count' #'density_(cells/mm^3)', 'cell_density'
    # drugA = 'PSI'
    # drugB = 'aPSI'
    # lightsheetDBAll = lightsheet_data

    # Identify regions in the data set, retrieve them.
    regionList = list(lightsheetDBAll.Region_Name.unique())
    regionSlices = np.array_split(np.asarray(regionList), 10)

    # Take the mean across each of the brain regions for each drug
    lightsheetDB = lightsheetDBAll.groupby(['abbreviation', 'Region_Name', 'Brain_Area', 'drug'])[
        dataColumn].mean().reset_index()

    # Calculate the mean for each drug
    lightsheetDB_PSI = lightsheetDB.query("drug == @drugA").reset_index()
    lightsheetDB_aPSI = lightsheetDB.query("drug == @drugB").reset_index()
    lightsheetDB_Diff = lightsheetDB_PSI.copy()
    lightsheetDB_Diff = lightsheetDB_Diff.rename(columns={dataColumn: drugA})
    lightsheetDB_Diff[drugB] = lightsheetDB_aPSI[dataColumn]

    # Calculate differences per region
    lightsheetDB_Diff['drug_diff'] = (
        lightsheetDB_Diff[drugA] - lightsheetDB_Diff[drugB]) / lightsheetDB_Diff[drugB] * 100
    lightsheetDB_Diff = lightsheetDB_Diff.sort_values(
        'drug_diff', ascending=False)
    brainAreaList = list(lightsheetDB_Diff.Brain_Area.unique())

    # Histogram time
    for brain_area_i, brain_area in enumerate(brainAreaList):
        dataSet = np.asarray(lightsheetDB_Diff.query(
            "Brain_Area == @brain_area")['drug_diff'])
        plt.plot(dataSet)
        plt.legend(brainAreaList)

    plt.xlabel('Region Index')
    plt.ylabel('Percent Difference (X-Y/Y * 100))')
    plt.ylim([-100, ylimMax])
    plt.title(drugA + ' vs ' + drugB + ' ' + dataColumn)
    plt.show()
