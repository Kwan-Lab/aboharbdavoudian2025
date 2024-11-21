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

def drug_stats_and_changes(databaseFrame, drugList, salSwitch):
    # Function for detecting changes in activation density across 2 drug conditions.
    # Returns a table containing a column representing 'cell_density_change' across conditions per region((X - Y)/Y * 100).
    # salSwitch - only generates saline - X drug comparisons if true, else returns all possible comparisons.
    # Identify drugs in the dataset

    drugCount = len(drugList)
    uniquePairCt = drugCount * (drugCount - 1) // 2

    drugDataFrames = np.empty(drugCount, dtype=object)
    drugStatsAll = np.empty(drugCount, dtype=object)
    
    # Gather Statistics for them individually
    for drug_i in range(0, drugCount):
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
    drugListInds = range(0, drugCount)
    drugPairIndList = [(a, b) for idx, a in enumerate(drugListInds) for b in drugListInds[idx + 1:]]

    if salSwitch:
        sal_idx = drugList.index('SAL')
        drugPairList = [pair for pair in drugPairList if 'SAL' in pair]
        drugPairIndList = [pair for pair in drugPairIndList if sal_idx in pair]
        

    drugPairDataList = []
    drugCompNames = [A + '-' + B for A, B in drugPairList]

    # Initialize DB which will have comparison info
    drugPairDB = databaseFrame.drop_duplicates(subset='Region_ID')[['Region_ID', 'abbreviation', 'Region_Name']]

    # for comp_i in range(0, uniquePairCt):
    for drugCompName, (drugA_idx, drugB_idx) in zip(drugCompNames, drugPairIndList):

        # Merge the two tables.
        drugX_drugY = pd.merge(drugDataFrames[drugA_idx], drugDataFrames[drugB_idx], on='Region_Name', how='inner')

        # Calculate cell_density_change
        drugX_drugY[drugCompName] = ((drugX_drugY['cell_density_x'] - drugX_drugY['cell_density_y']) / drugX_drugY['cell_density_y']) * 100

        # Clean up the table
        drugX_drugY = drugX_drugY.drop(['dataset_x', 'drug_x', 'cell_density_x', 'dataset_y', 'drug_y', 'cell_density_y'], axis=1)
        drugX_drugY = drugX_drugY.rename(columns={'abbreviation_x': 'abbreviation', 'Region_ID_x': 'Region_ID'})


        # Store in array, mainly to calculate error bars on differences in Figure 2. Putting out single Dataframe overwhelms memory (32 GB at least)
        # Additional Note: lists can't be merged due to different numbers in each drug group.
        drugPairDataList.append(drugX_drugY)
        drugX_drugY_meanPerRegion = drugX_drugY.groupby(['Region_ID', 'abbreviation', 'Region_Name']).mean().reset_index()

        # Combine with data up to this point
        drugPairDB = pd.merge(drugPairDB, drugX_drugY_meanPerRegion[['Region_ID', drugCompName]], on='Region_ID', how='inner')


    # Clean up final table as needed
    # #remove duplicate column names
    drugPairDB = drugPairDB.loc[:, ~drugPairDB.columns.duplicated()]

    # Identify regions with infs/nans/etc
    regions2Remove = drugPairDB.Region_Name[drugPairDB.isin([np.nan, np.inf, -np.inf]).any(1)]

    # Remove these regions from all output structures
    drugPairDB = drugPairDB[~drugPairDB['Region_Name'].isin(regions2Remove)]
    drugStatsAll = drugStatsAll[~drugStatsAll['Region_Name'].isin(regions2Remove)]
    for drugComp_i, drugComp in enumerate(drugPairDataList):
        drugPairDataList[drugComp_i] = drugComp[~drugComp['Region_Name'].isin(regions2Remove)]

    return drugPairDB, drugStatsAll, drugPairDataList, drugCompNames

def brainRegionTrend(lightsheetDBAll, dataColumn, drugA, drugB, ylimMax=0):
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
    lightsheetDB_Diff['drug_diff'] = (lightsheetDB_Diff[drugA] - lightsheetDB_Diff[drugB]) / lightsheetDB_Diff[drugB] * 100
    lightsheetDB_Diff = lightsheetDB_Diff.sort_values('drug_diff', ascending=False)
    brainAreaList = list(lightsheetDB_Diff.Brain_Area.unique())

    # Histogram time
    for brain_area_i, brain_area in enumerate(brainAreaList):
        dataSet = np.asarray(lightsheetDB_Diff.query(
            "Brain_Area == @brain_area")['drug_diff'])
        plt.plot(dataSet)
        plt.legend(brainAreaList)

    plt.xlabel('Region Index')
    plt.ylabel('Percent Difference (X-Y/Y * 100))')
    if ylimMax != 0:
        plt.ylim([-100, ylimMax])

    plt.title(drugA + ' vs ' + drugB + ' ' + dataColumn)
    plt.show()

def collect_CI(cfos_diff, cfos_diff_labels, drugList, dirDict, batchSplit):

    # Define Paths
    tmpFileName = 'drug_ci_range_db_utils.h5.npy'
    tempDataFilename = os.path.join(dirDict['tempDir'], tmpFileName)

    if os.path.exists(tempDataFilename):
        print('Loading ' + tmpFileName + '...')

        # Load previous saved file
        ciRangeDB = pd.read_pickle(tempDataFilename)

        return ciRangeDB
    
    print('Generating ' + tmpFileName + '...')
    
    if batchSplit:
        # More complicated to focus on aSAL vs SAL for the right drugs.
        drugListA = [x for x in drugList if x[0] == 'a']
        drugListB = [x for x in drugList if x[0] != 'a']
        drugPairListA = [(a + '-'+ b) for idx, a in enumerate(drugListA) for b in drugListA[idx + 1:]]
        drugPairListB = [(a + '-'+ b) for idx, a in enumerate(drugListB) for b in drugListB[idx + 1:]]
        drugPairList = drugPairListA + drugPairListB

        # Brief additional processing to make sure Batch 1 is processed w/ batch 1 saline (aSAL)
        salListA = [i for i in drugPairListA if 'SAL' in i]
        salListB = [i for i in drugPairListB if 'SAL' in i]
        salList = salListA + salListB

    else:
        drugPairList = [(a +'-'+ b) for idx, a in enumerate(drugList) for b in drugList[idx + 1:]]
        salList = [i for i in drugPairList if 'SAL' in i]

    for idx, drugPair in enumerate(tqdm(salList)):
        
        dataInd = list(cfos_diff_labels).index(drugPair)

        # Make life easier - collect needed dots
        regionList = cfos_diff[dataInd].Region_Name.unique()
        dataCol = cfos_diff_labels[dataInd]
        dataTable = cfos_diff[dataInd]
        ciData = np.empty([len(regionList), 2])

        # Two methods for determining CIs. Method 2 is faster, and seems to yield identical responses.
        methodSwitch = 2

        if methodSwitch == 1:
        # Method 1 - Sure about region/number match - very slow.
            for region_i, region in enumerate(regionList):
                regionData = dataTable.loc[dataTable.Region_Name == region][dataCol]
                ciData[region_i, :] = sns.utils.ci(sns.algorithms.bootstrap(regionData), which=95)

            if idx == 0:
                ciRangeDB = pd.DataFrame()
                ciRangeDB['Region_Name'] = regionList

            ciRangeDB[drugPair + '_lower'] = ciData[:,0]
            ciRangeDB[drugPair + '_upper'] = ciData[:,1]

        elif methodSwitch == 2:
        # Method 2 - Plot it, things are done faster.
            # Figure
            plt.figure(figsize=(20,160))

            ax = sns.pointplot(y='Region_Name', x=dataCol, data = dataTable, errorbar=('ci', 95), join=False, units=16, errwidth = 0.5, color='red',dodge=True)

            # Pull values
            if idx == 0:
                ciRangeDB = pd.DataFrame()
                ciRangeDB['Region_Name'] = [textObj.get_text() for textObj in ax.axes.get_yticklabels()]

            ciRangeDB[drugPair + '_upper'] = [line.get_xdata().max() for line in ax.lines]
            ciRangeDB[drugPair + '_lower'] = [line.get_xdata().min() for line in ax.lines]
            plt.clf()

    # Save for later use
    ciRangeDB.to_pickle(tempDataFilename)
    
    return ciRangeDB

def gen_gene_corr(cfos_x_genes, splitTag, drugListActive, regionSet, dirDict):

    brainAreaList = list(cfos_x_genes['Brain Area'].unique())

    list_of_genes = cfos_x_genes.gene_acronym.unique()

    # List of genes shortening for testing loop
    # list_of_genes = list_of_genes[0:100]

    gene_count = len(list_of_genes)
    gene_drug_corr = np.empty([len(drugListActive), len(regionSet)], dtype=object)

    for set_i, set_name in enumerate(regionSet):
        for drug_i, drug in enumerate(drugListActive):

            all_correlations = np.empty([gene_count])
            tempDataFilename = os.path.join(dirDict['geneCorrDir'], f'{drug}_{set_name}_corr_db.h5')

            if os.path.exists(tempDataFilename):
                print(f'{set_name} {drug} already generated...')

                # Load previous saved file
                # all_correlations = pd.read_pickle(tempDataFilename)
                
            else:
                print(f'Processing {set_name} {drug}...')
                
                if drug[0] == splitTag[0]:
                    drugCol = drug + '-' + 'aSAL'
                else:
                    drugCol = drug + '-SAL'

                # Based on the desired comparison, use different datasets
                if set_name == 'Whole':
                    cfos_data_genes = cfos_x_genes.copy()
                elif set_name in brainAreaList:
                    cfos_data_genes = cfos_x_genes[cfos_x_genes['Brain Area'] == set_name].copy()
                elif set_name == 'Top_25':
                    cfos_data_genes = cfos_x_genes.copy()
                elif set_name == 'Cortex_Top_25':
                    cfos_data_genes = cfos_x_genes[cfos_x_genes['Brain Area'] == 'Cortex'].copy()
                else:
                    raise ValueError('Invalid region set name. Make sure to match case of Brain Area')

                # Iterate across genes, finding correlations
                grouped_data = cfos_data_genes.groupby('gene_acronym')

                for gene_i, gene in enumerate(tqdm(list_of_genes)):
                    if gene in grouped_data.groups:  # Check if gene exists in the grouped data
                        cfos_x_genes_of_interest = grouped_data.get_group(gene)
                        # Vectorized correlation computation

                        if 'Top_25' in set_name:
                            zScoreThres = np.percentile(cfos_x_genes_of_interest.zscore, 75)
                            cfos_x_genes_of_interest = cfos_x_genes_of_interest[cfos_x_genes_of_interest.zscore >= zScoreThres]

                        correlation = cfos_x_genes_of_interest['zscore'].corr(cfos_x_genes_of_interest[drugCol])
                        all_correlations[gene_i] = correlation
                    else:
                        all_correlations[gene_i] = None  # If gene doesn't exist, store None or other placeholder

                all_correlations = pd.DataFrame(np.column_stack((list_of_genes, all_correlations)), columns=['gene', drug + ' correlation'])

                colData = all_correlations[drug + ' correlation']
                all_correlations['percentile'] = colData.rank(pct = True)
                all_correlations = all_correlations.sort_values(by=['percentile'], ascending=False)

                # Save for later use
                all_correlations.to_pickle(tempDataFilename)  
                
            # gene_drug_corr[drug_i, set_i] = all_correlations
    
    # return gene_drug_corr