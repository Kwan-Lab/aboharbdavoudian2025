import os
import sys
import pandas as pd
import numpy as np
from os.path import exists, join
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time


def loadLightSheetData(dirDict, switchDict):
    # A function which loads lightsheet data from specifiied files.
    # Input includes path to files, alongside numbers defining which code is required to preprocess it.

    print('Data being loaded...')
    startTime = time.time()

    # Create a path for the intermediate to be saved. If it is present, load it. If not, create it.

    intFile = dirDict['tempDir'] + 'lightSheet_all.pkl'

    if not os.path.exists(intFile):

        ############## Batch 1 ##############
        # Merging first Batch of Light sheet data
        csv_list = ['sal_f1', 'sal_f2', 'sal_m1', 'sal_m2', 'ket_f1',
                    'ket_f2', 'ket_m1', 'ket_m2', 'psi_f1', 'psi_f2', 'psi_m1', 'psi_m2']
        csv_drug_list = [x.split('_')[0].upper() for x in csv_list]
        sex_list = [x.split('_')[1][0].upper() for x in csv_list]

        if switchDict['batchSplit']:
            csv_drug_list = [switchDict['splitTag']
                             [0] + x for x in csv_drug_list]

        B1_dataset_list = ['SAL5', 'SAL6', 'SAL7', 'SAL8', 'KET1',
                           'KET2', 'KET3', 'KET4', 'PSI5', 'PSI6', 'PSI7', 'PSI8']

        for csv_i, csvName in enumerate(csv_list):
            tmpDb = pd.read_csv(dirDict['B1'] + csvName + '.csv', sep=',')

            tmpDb['drug'] = csv_drug_list[csv_i]
            tmpDb['sex'] = sex_list[csv_i]
            tmpDb['dataset'] = B1_dataset_list[csv_i]
            tmpDb['total_cells'] = (tmpDb.iloc[2]['count'] + tmpDb.iloc[840]['count'])

            if csv_i == 0:
                lightSheet_B1 = tmpDb
            else:
                lightSheet_B1 = pd.concat([lightSheet_B1, tmpDb])

        # remove, prepare data
        background_ROIs = ['background', 'left root', 'right root']
        lightSheet_B1 = lightSheet_B1[~lightSheet_B1['name'].isin(background_ROIs)]

        # remove unneccesary columns
        lightSheet_B1 = lightSheet_B1.drop(['acronym', 'parent_structure_id', 'depth', 'density (cells/mm^3)'], axis=1)

        # rename ID column correctly
        lightSheet_B1 = lightSheet_B1.rename(columns={'id': 'graph_order', 'name': 'Region'})

        ############## Batch 2 ##############
        if switchDict['oldBatch2']:
            lightSheetRawData = pd.read_excel(
                dirDict['B2_Orig'] + 'AlexKwan_40brainproject_NeuNcFos 642 density.xlsx', sheet_name='Raw Data')
        else:
            lightSheetRawData = pd.read_excel(
                dirDict['B2'] + 'AlexKwan_40brainproject_NeuNcFos 642 density_V2.xlsx', sheet_name='All Samples')

            # In the #Realigned data, Region was swapped to 'name'. Change it back.
        lightSheetRawData = lightSheetRawData.rename(
            columns={'name': 'Region'})

        # - Pull Columns, detect drug related columns.
        drugColumns = lightSheetRawData.columns[lightSheetRawData.columns.str.contains(
            'AlKw') * lightSheetRawData.columns.str.contains('count')]  # Columns w/ AlKw are drug related

        for drug_col_i, drug_col in enumerate(drugColumns):

            densityCol = drug_col.replace(' count', ' density (cells/mm^3)')

            # Pull out the counts and the region columns
            tmpDb = pd.concat([lightSheetRawData['Region'],
                               lightSheetRawData[drug_col], lightSheetRawData[densityCol]], axis=1)
            tmpDb = tmpDb.applymap(
                lambda x: x.strip() if isinstance(x, str) else x)

            # Remove prefixes  if present
            tmpString = drug_col.replace("AlKw_", "").replace(
                "52022_", "").replace(" count", "")

            # Correct the A_ and C_ to A- and C-
            tmpString = tmpString.replace(
                "C_SSRI", "C-SSRI").replace("A_SSRI", "A-SSRI").replace('__', '_').replace('_ ', '_')

            # There is a typo in the Data sheet - 6DET should be 6FDET (only true for M2)
            tmpString = tmpString.replace("6DET", "6FDET")

            # Create the table                                                              # Table starts w/ Region and count
            drug = tmpString.split('_')[0]
            # drug - sal, psi, ket
            tmpDb['drug'] = drug
            # sex - m, f
            tmpDb['sex'] = tmpString.split('_')[1]
            # dataset - [drug][#]
            tmpDb['dataset'] = drug + tmpString.split('_')[2][0]
            # total_cells - total cells across all regions in animal.
            tmpDb = tmpDb.rename(
                columns={drug_col: 'count', densityCol: 'density (cells/mm^3)'})
            tmpDb['total_cells'] = int(tmpDb.loc[tmpDb['Region'] == 'left root', 'count']) + int(
                tmpDb.loc[tmpDb['Region'] == 'right root', 'count'])

            if drug_col_i == 0:
                lightSheet_B2 = tmpDb
            else:
                lightSheet_B2 = pd.concat([lightSheet_B2, tmpDb])

            # remove, prepare data
            background_ROIs = ['background', 'left root', 'right root']
            lightSheet_B2 = lightSheet_B2[~lightSheet_B2['Region'].isin(
                background_ROIs)]

        if switchDict['testSplit']:
            # A list of datasets to artifically shift to a 'new drug' for the sake of testing within drug variability.
            drugDataSet = 'C-SSRI1', 'C-SSRI2', 'DMT1', 'DMT2'
            newDrugName = 'C-SSRI_t', 'C-SSRI_t', 'DMT_t', 'DMT_t'

            # Cycle through each drug in the list above, change it
            for drug_i, drug in enumerate(drugDataSet):
                swapInd = lightSheet_B2['dataset'] == drug
                lightSheet_B2['drug'][swapInd] = newDrugName[drug_i]

        ############## Batch 3 ##############
        # MDMA Batch 3
        lightSheetRawDataB3 = pd.read_excel(
            dirDict['B3'] + 'AlexKwan_12brainproject_NeuNcFos 642 density.xlsx', sheet_name='All Samples')

        lightSheetRawDataB3 = lightSheetRawDataB3.rename(
            columns={'name': 'Region'})

        rawColNames = lightSheetRawDataB3.columns

        # Process the Titles to be appropriate names...
        repStrings = [
            ("C1M2", "MDMA_M_1"),
            ("C1M3", "MDMA_M_2"),
            ("C2F2", "MDMA_F_8"),
            ("C3M5", "MDMA_M_3"),
            ("C4F5", "MDMA_F_5"),
            ("C2F3", "MDMA_F_6"),
            ("C3M4", "MDMA_M_4"),
            ("C4F4", "MDMA_F_7")]

        if switchDict['batchSplit']:
            repStrings = repStrings + [("k1m1", "cKET_M_1"),
                                       ("k2m2", "cKET_M_2"),
                                       ("k3f1", "cKET_F_3"),
                                       ("k4f2", "cKET_F_4")]
        else:
            repStrings = repStrings + [("k1m1", "KET_M_5"),
                                       ("k2m2", "KET_M_6"),
                                       ("k3f1", "KET_F_7"),
                                       ("k4f2", "KET_F_8")]

        for old, new in repStrings:
            rawColNames = rawColNames.str.replace(old, new)

        lightSheetRawDataB3.columns = rawColNames

        # - Pull Columns, detect drug related columns.
        drugColumns = lightSheetRawDataB3.columns[lightSheetRawDataB3.columns.str.contains(
            'Kwan') * lightSheetRawDataB3.columns.str.contains('count')]

        for drug_col_i, drug_col in enumerate(drugColumns):

            densityCol = drug_col.replace(' count', ' density (cells/mm^3)')

            # Pull out the counts and the region columns
            tmpDb = pd.concat([lightSheetRawDataB3['Region'],
                              lightSheetRawDataB3[drug_col], lightSheetRawDataB3[densityCol]], axis=1)
            tmpDb = tmpDb.applymap(
                lambda x: x.strip() if isinstance(x, str) else x)

            # Remove prefixes  if present
            tmpString = drug_col.replace("Kwan_120222_", "").replace(
                "52022_", "").replace(" count", "")

            # Create the table                                                              # Table starts w/ Region and count
            drug = tmpString.split('_')[0]
            # drug - sal, psi, ket
            tmpDb['drug'] = drug
            # sex - m, f
            tmpDb['sex'] = tmpString.split('_')[1]
            # dataset - [drug][#]
            tmpDb['dataset'] = drug + tmpString.split('_')[2]
            # total_cells - total cells across all regions in animal.
            tmpDb = tmpDb.rename(columns={drug_col: 'count', densityCol: 'density (cells/mm^3)'})
            tmpDb['total_cells'] = int(tmpDb.loc[tmpDb['Region'] == 'left root', 'count']) + int(tmpDb.loc[tmpDb['Region'] == 'right root', 'count'])

            if drug_col_i == 0:
                lightSheet_B3 = tmpDb
            else:
                lightSheet_B3 = pd.concat([lightSheet_B3, tmpDb])

        # remove, prepare data
        background_ROIs = ['background', 'left root', 'right root']
        lightSheet_B3 = lightSheet_B3[~lightSheet_B3['Region'].isin(
            background_ROIs)]
        lightSheet_B3.to_csv(dirDict['debugDir'] + 'lightsheet_B3.csv')

        ############## Merge All the Datasets ##############
        # Create a dict for B1's graph order to add it to the B2.
        graphOrder = list(lightSheet_B1.graph_order)
        nameOrder = list(lightSheet_B1.Region)
        graphDict = dict(zip(nameOrder, graphOrder))

        # Create graph_order in B2/B3
        lightSheet_B2['graph_order'] = lightSheet_B2.Region.map(graphDict)
        lightSheet_B3['graph_order'] = lightSheet_B3.Region.map(graphDict)

        # Convert Density to volume in each
        lightSheet_B2['volume (mm^3)'] = lightSheet_B2['count'] / lightSheet_B2['density (cells/mm^3)']
        lightSheet_B2 = lightSheet_B2.drop(columns=['density (cells/mm^3)'])

        lightSheet_B3['volume (mm^3)'] = lightSheet_B3['count'] / lightSheet_B3['density (cells/mm^3)']
        lightSheet_B3 = lightSheet_B3.drop(columns=['density (cells/mm^3)'])

        # 4 regions don't exist in the old data (left/right dorsal/ventral hippocampus)
        setA = set(lightSheet_B1.Region)
        setB = set(lightSheet_B2.Region)
        setC = set(lightSheet_B3.Region)

        # Remove them from B and C
        roi_to_remove = setA.difference(setB)
        lightSheet_B2 = lightSheet_B2[~lightSheet_B2['Region'].isin(roi_to_remove)]
        roi_to_remove = setA.difference(setC)
        lightSheet_B3 = lightSheet_B3[~lightSheet_B3['Region'].isin(roi_to_remove)]

        # Merge the two table
        # if includeBatch3:
        lightSheet_all = pd.concat([lightSheet_B1, lightSheet_B2, lightSheet_B3])
        # else:
        #     lightSheet_all = pd.concat([lightSheet_B1, lightSheet_B2])

        # Debug Stop A - Post Merging of the datasets
        if switchDict['debugOutputs']:
            debugReport(lightSheet_all, 'lightSheet_all', dirDict['debug_outPath'])

        ############## Merge Left and Right sides of the Datasets ##############

        # Subtract 10000 from the graph order of the right side of the brain
        lightSheet_all.loc[lightSheet_all['Region'].str.startswith('right'), 'graph_order'] -= 10000

        # Merge left and right sides by summing their counts and volumes.
        lightSheet_all = lightSheet_all.groupby(['dataset', 'graph_order', 'sex', 'drug', 'total_cells'], as_index=False).sum().copy()

        # recalculate density with left and right sides combined
        lightSheet_all['density (cells/mm^3)'] = (lightSheet_all.loc[:, 'count'] / lightSheet_all.loc[:, 'volume (mm^3)'])

        ############## convert 'graph_order' to Region_ID and Region_Name, per Atlas ##############
        # remap IDs to match with ABA
        # allen brain structure tree
        ABA_tree = pd.read_csv(dirDict['atlasDir'] + 'structure_tree_2017.csv')

        # map new atlas_id onto graph_id
        lightSheet_all['Region ID'] = lightSheet_all.graph_order.map(
            ABA_tree.set_index('graph_order')['id'].to_dict())
        # map new names onto graph_id
        lightSheet_all['Region Name'] = lightSheet_all.graph_order.map(
            ABA_tree.set_index('graph_order')['name'].to_dict())

        # remove unneccesary ID columns to avoid confusion
        lightSheet_all = lightSheet_all.drop(['graph_order'], axis=1)

        # reorder columns
        lightSheet_all = lightSheet_all[['Region ID', 'Region Name', 'count',
                                         'volume (mm^3)', 'density (cells/mm^3)', 'sex', 'drug', 'dataset', 'total_cells']]

        ############## Scale the density/counts ##############

        # #scaling factor is drug avg / animal total cell'
        lightsheet_data = lightSheet_all.copy()
        lightsheet_data['scaling_factor'] = 1

        # lightsheet_data = lightsheet_all.merge(drug_means, how='left', left_on=['drug'], right_on=['drug'])
        if switchDict['scalingFactor']:

            # Identify Batch 1
            scaleRows = lightsheet_data.dataset.isin(B1_dataset_list)

            # Divide by the total cells across the drug group, multiple by mean across that group in the subsequent batch.
            drugsInGroup = ['KET', 'PSI', 'SAL']

            # Create the scaling factor per drug - mean_drug_NewBatch/mean_drug_Batch1
            for drug in drugsInGroup:
                # Identify the specific drugs rows to be modified
                drug_scaleRows = scaleRows * lightsheet_data.dataset.str.contains(drug)
                drug_ref_scaleRows = ~scaleRows * lightsheet_data.dataset.str.contains(drug)

                # Scale these rows by [Drug Mean New Batch]/[Drug Mean Old Batch]
                lightsheet_data.loc[drug_scaleRows, 'scaling_factor'] = np.mean(lightsheet_data.total_cells[drug_ref_scaleRows])/np.mean(lightsheet_data.total_cells[drug_scaleRows])

        # Scale counts and total counts
        lightsheet_data['count'] = np.round(lightsheet_data['count'] * lightsheet_data['scaling_factor'])
        lightsheet_data['total_cells'] = np.round(lightsheet_data['total_cells'] * lightsheet_data['scaling_factor']) # This will lead to instances where total_cells != sum(all cells). by a few.
        lightsheet_data['cell_density'] = np.round(lightsheet_data['count'] / lightsheet_data['volume (mm^3)'], 2)
        
        # Fill Nans
        lightsheet_data = lightsheet_data.fillna(0)

        ############## Merging Additional Atlas Data ##############

        # get allen brain atlases
        ABA_tree = pd.read_csv(dirDict['atlasDir'] + 'ABA_CCF.csv')
        ABA_tree = ABA_tree.rename(columns={
                                   'structure ID': 'id', '"Summary Structure" Level for Analyses': 'summary_struct'})
        ABA_tree = ABA_tree.drop(columns=['order', 'parent_id', 'depth in tree', 'structure_id_path', 'total_voxel_counts (10 um)',
                                          'Structure independently delineated (not merged to form parents)'])

        ABA_hier = pd.read_csv(dirDict['atlasDir'] + 'ABAHier2017_csv.csv')
        ABA_hier = ABA_hier.loc[:, ~ABA_hier.columns.str.contains('^Unnamed')]
        ABA_hier = ABA_hier.rename(columns={'Region ID': 'id', 'CustomRegion': 'Brain Area'})

        # merge the atlases
        ABA_tree = pd.merge(ABA_tree, ABA_hier, on=['id'])

        # define hierarchy by 'summary structure' from Wang/Ding . . . Ng, Cell 2020
        hierarchy_meso = ABA_tree[ABA_tree.summary_struct == 'Y']
        # hierarchy_meso = ABA_tree
        
        # remove ventricular systems, fiber tracts
        remove_list = ['Fibretracts', 'VentricularSystems']
        hierarchy_meso_ids = hierarchy_meso[~hierarchy_meso['Brain Area'].isin(remove_list)]

        # cleaning up and Merging with Data
        hierarchy_meso_ids = hierarchy_meso_ids.rename(columns={'id': 'Region ID'})
        hierarchy_meso_ids = hierarchy_meso_ids.drop(columns=['full structure name', 'Major Division', 'summary_struct'])

        lightsheet_data = pd.merge(lightsheet_data, hierarchy_meso_ids, on=['Region ID'])
        # lightsheet_data.to_pickle('testScripts//lightsheet_data_big.pkl')

        # Rename Columns
        cols2 = ['Region ID', 'abbreviation', 'Region Name', 'count',
                 'volume (mm^3)', 'density (cells/mm^3)', 'sex', 'drug', 'dataset', 'cell_density', 'Brain Area', 'total_cells']
        lightsheet_data = lightsheet_data[cols2]
        lightsheet_data = lightsheet_data.rename(columns={'Region ID': 'Region_ID', 'Region Name': 'Region_Name',
                                                 'volume (mm^3)': 'volume_(mm^3)', 'density (cells/mm^3)': 'density_(cells/mm^3)', 'Brain Area': 'Brain_Area'})
        

        # Drop 'density_(cells/mm^3)'
        lightsheet_data.drop(['density_(cells/mm^3)'], axis=1, inplace=True)

        debug_ROI = 'Dorsal Raphe'

        if switchDict['debugOutputs']:
            debugReport(lightsheet_data, 'lightsheet_data',dirDict['debug_outPath'], 'Region_Name', debug_ROI)

        ############## Saving ##############

        print('Done, Saving file...')
        lightsheet_data.to_pickle(intFile)

    else:
        # Just load it and return it
        lightsheet_data = pd.read_pickle(intFile)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(round(executionTime, 2)))

    return lightsheet_data


def createDirs(rootDir, switchDict, dirDict):

    if switchDict['testSplit'] or switchDict['batchSplit'] or switchDict['scalingFactor'] or switchDict['oldBatch2']:
        # tsplitTag = dsplitTag = b3tag = scaleTag = ''
        tsplitTag = dsplitTag = scaleTag = batch2Tag = ''

        if switchDict['testSplit']:
            tsplitTag = 'testSplit'

        if switchDict['batchSplit']:
            dsplitTag = 'split'

        if switchDict['scalingFactor']:
            scaleTag = 'scaled'

        if switchDict['oldBatch2']:
            batch2Tag = 'oldB2'

        # if includeBatch3:
        #     b3tag = 'B3'

        # stringVar = (tsplitTag, dsplitTag, tsplitTag, b3tag)
        stringVar = (scaleTag, dsplitTag, tsplitTag, batch2Tag)
        stringVar = [i for i in stringVar if i]
        dirString = str(len(stringVar)) + '.' + '_'.join(stringVar) + '_'

    else:
        dirString = '0._'

    tempDir = rootDir + dirString + 'Temp\\'
    outDir = rootDir + dirString + 'Output\\'
    classifyDir = join(outDir, 'classif\\')
    debugDir = rootDir + dirString + 'Debug\\'  # Debugging paths and setup
    debug_outPath = debugDir + 'lightSheet_all_ROI.xlsx'

    # Make directories if they don't exist
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)

    if not os.path.exists(outDir):
        os.mkdir(outDir)

    if not os.path.exists(debugDir):
        os.mkdir(debugDir)

    if not os.path.exists(classifyDir):
        os.mkdir(classifyDir)

    dirDict['debugDir'] = debugDir
    dirDict['tempDir'] = tempDir
    dirDict['outDir'] = outDir
    dirDict['classifyDir'] = classifyDir
    dirDict['debug_outPath'] = debug_outPath

    return dirDict

def debugReport(pdDataFrame, sheetName, debug_outPath, roiColName, debug_ROI):
    if len(debug_ROI) == 0:
        if os.path.exists(debug_outPath):
            with pd.ExcelWriter(debug_outPath, mode='a', if_sheet_exists='replace') as writer:
                pdDataFrame.to_excel(writer, sheetName)
        else:
            with pd.ExcelWriter(debug_outPath) as writer:
                pdDataFrame.to_excel(writer, sheetName)
    else:

        for Roi_i, Roi in enumerate(debug_ROI):
            tmp = pdDataFrame.loc[pdDataFrame[roiColName].str.contains(
                debug_ROI[Roi_i])]
            if Roi_i == 0:
                tmp_out = tmp
            else:
                tmp_out = pd.concat([tmp_out, tmp])

        # Export the filtered list
        if os.path.exists(debug_outPath):
            with pd.ExcelWriter(debug_outPath, mode='a', if_sheet_exists='replace') as writer:
                tmp_out.to_excel(writer, sheetName + '_ROI')
        else:
            with pd.ExcelWriter(debug_outPath) as writer:
                tmp_out.to_excel(writer, sheetName + '_ROI')
