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

    if 1: #not os.path.exists(intFile):

        # Load batch 1        
        lightSheet_B1 = load_lightsheet_batchCSV(dirDict, switchDict, 'B1')

        # Load batch 2        
        excelfilePath_B2 = dirDict['B2'] + 'AlexKwan_40brainproject_NeuNcFos 642 density_V2.xlsx'
        lightSheet_B2 = load_lightsheet_excel(excelfilePath_B2, dirDict, switchDict, 'B2')

        # Load batch 3
        excelfilePath_B3 = dirDict['B3'] + 'density channel 642.xlsx'
        lightSheet_B3 = load_lightsheet_excel(excelfilePath_B3, dirDict, switchDict, 'B3')

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

        lightSheet_all = pd.concat([lightSheet_B1, lightSheet_B2, lightSheet_B3])

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
        lightSheet_all['Region ID'] = lightSheet_all.graph_order.map(ABA_tree.set_index('graph_order')['id'].to_dict())
        # map new names onto graph_id
        lightSheet_all['Region Name'] = lightSheet_all.graph_order.map(ABA_tree.set_index('graph_order')['name'].to_dict())

        # remove unneccesary ID columns to avoid confusion
        lightSheet_all = lightSheet_all.drop(['graph_order'], axis=1)

        # reorder columns, drop graph_order
        lightSheet_all = lightSheet_all[['Region ID', 'Region Name', 'count', 'volume (mm^3)', 'density (cells/mm^3)', 'sex', 'drug', 'dataset', 'total_cells']]

        if switchDict['debugOutputs']:
            lightSheet_all.to_csv(dirDict['debugDir'] + 'lightsheet_atlas_all.csv')

        ############## Scale the density/counts ##############

        # #scaling factor is drug avg / animal total cell'
        lightsheet_data = lightSheet_all.copy()
        lightsheet_data['scaling_factor'] = 1

        # lightsheet_data = lightsheet_all.merge(drug_means, how='left', left_on=['drug'], right_on=['drug'])
        if switchDict['scalingFactor']:

            # Identify Batch 1
            B1_dataset_list = lightSheet_B1.dataset.unique()
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
        lightsheet_data['total_cells'] = np.round(lightsheet_data['total_cells'] * lightsheet_data['scaling_factor'])
        lightsheet_data['cell_density'] = np.round(lightsheet_data['count'] / lightsheet_data['volume (mm^3)'], 2)
        
        # Fill Nans
        lightsheet_data = lightsheet_data.fillna(0)

        # Convert some drug names to desired names in the dataset
        lightsheet_data['drug'] = lightsheet_data['drug'].replace('DMT', '5MEO')
        lightsheet_data['drug'] = lightsheet_data['drug'].replace('6FDET', '6-F-DET')

        ############## Merging Additional Atlas Data ##############

        # get Allen brain atlases
        ABA_tree = pd.read_csv(dirDict['atlasDir'] + 'ABA_CCF.csv')
        ABA_tree = ABA_tree.rename(columns={'structure ID': 'id', '"Summary Structure" Level for Analyses': 'summary_struct'})
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
        
        # Exclude Fibre Tracts and Ventricular Systems
        remove_list = ['Fibretracts', 'VentricularSystems']
        lightsheet_data = lightsheet_data[~lightsheet_data['Brain_Area'].isin(remove_list)]

        # Drop 'density_(cells/mm^3)'
        lightsheet_data.drop(['density_(cells/mm^3)'], axis=1, inplace=True)

        # Add a normalized version of counts for later classification
        lightsheet_data.loc[:, 'count_norm'] = lightsheet_data.loc[:, 'count']/lightsheet_data.loc[:, 'total_cells']
        lightsheet_data.loc[:, 'density_norm'] = lightsheet_data.loc[:, 'count_norm']/lightsheet_data.loc[:, 'volume_(mm^3)']

        lightsheet_data = lightsheet_data.fillna(0)

        lightsheet_data.to_csv(dirDict['debugDir'] + 'lightsheet_atlas_summary.csv')

        # Overwrite the dataset values with the updated drug names.
        lightsheet_data['dataset'] = lightsheet_data['drug'] + lightsheet_data['dataset'].str[-1]

        ############## Convert the drug names from strings to categorical variables in sequence ##############
        customOrder = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']
        lightsheet_data['drug'] = pd.Categorical(lightsheet_data['drug'], categories=customOrder, ordered=True)

        if switchDict['debugOutputs']:
            debug_ROI = 'Dorsal Raphe' 
            debugReport(lightsheet_data, 'lightsheet_data', dirDict['debug_outPath'], 'Region_Name', debug_ROI)

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

    if switchDict['testSplit'] or switchDict['batchSplit'] or switchDict['scalingFactor']:
        # tsplitTag = dsplitTag = b3tag = scaleTag = ''
        tsplitTag = dsplitTag = scaleTag = batch2Tag = ''

        if switchDict['testSplit']:
            tsplitTag = 'testSplit'

        if switchDict['batchSplit']:
            dsplitTag = 'split'

        if switchDict['scalingFactor']:
            scaleTag = 'scaled'

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

def load_lightsheet_batchCSV(dirDict, switchDict, debugTag):
    # Merging first Batch of Light sheet data

        # Hard coded elements
        csv_list = ['sal_f1', 'sal_f2', 'sal_m1', 'sal_m2', 'ket_f1', 'ket_f2', 'ket_m1', 'ket_m2', 'psi_f1', 'psi_f2', 'psi_m1', 'psi_m2']
        B1_dataset_list = ['SAL5', 'SAL6', 'SAL7', 'SAL8', 'KET1', 'KET2', 'KET3', 'KET4', 'PSI5', 'PSI6', 'PSI7', 'PSI8']

        # Extract information from the title names
        csv_drug_list = [x.split('_')[0].upper() for x in csv_list]
        sex_list = [x.split('_')[1][0].upper() for x in csv_list]

        # If you want to examine batches, append a tag to the drug name.
        if switchDict['batchSplit']:
            csv_drug_list = [switchDict['splitTag'][0] + x for x in csv_drug_list]

        for csv_i, csvName in enumerate(csv_list):
            tmpDb = pd.read_csv(dirDict['B1'] + csvName + '.csv', sep=',')

            tmpDb['drug'] = csv_drug_list[csv_i]
            tmpDb['sex'] = sex_list[csv_i]
            tmpDb['dataset'] = B1_dataset_list[csv_i]
            tmpDb['total_cells'] = tmpDb[tmpDb['name'].str.contains('Basic cell groups')]['count'].sum().astype(int)
            tmpDb['count'] = tmpDb['count'].astype(int)

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

        if switchDict['debugOutputs']:
            lightSheet_B1.to_csv(dirDict['debugDir'] + f'lightsheet_{debugTag}.csv')
        
        return lightSheet_B1

def load_lightsheet_excel(excelFileName, dirDict, switchDict, dbFileTag):

    ls_Raw = pd.read_excel(excelFileName, sheet_name='All Samples')
    ls_Raw = ls_Raw.rename(columns={'name': 'Region'})

    # - Pull Columns, detect drug related columns.
    lsCol = ls_Raw.columns
    if dbFileTag == 'B2':
        drugCol_Orig = lsCol[lsCol.str.contains('AlKw') * lsCol.str.contains('count')]
        drugCol = [x.replace("AlKw_", "").replace("52022_", "").replace(" count", "").replace('_NeuN_cFos', '').replace('_NeuN_cFOS', '') for x in drugCol_Orig]
    elif dbFileTag == 'B3':
        drugCol_Orig = lsCol[lsCol.str.contains('Kwan') * lsCol.str.contains('count')]
        drugCol = [x.replace("Kwan_120222_", "").replace(" count", "") for x in drugCol_Orig]

    # Some dataset specific changes
    if dbFileTag == 'B2':
        # Replace some drug names
        drugCol = [x.replace("C_SSRI", "C-SSRI").replace("A_SSRI", "A-SSRI").replace("6DET", "6FDET").replace('__', '_').replace('_ ', '_') for x in drugCol]
    elif dbFileTag == 'B3':

        if switchDict['batchSplit']:
            sTag = switchDict['splitTag'][2]
            repStrings = [("KET_M_1", f"{sTag}KET_M_1"),
                          ("KET_M_2", f"{sTag}KET_M_2"),
                          ("KET_F_3", f"{sTag}KET_F_3"),
                          ("KET_F_4", f"{sTag}KET_F_4")]
        else:
            repStrings = [("KET_M_1", "KET_M_5"),
                          ("KET_M_2", "KET_M_6"),
                          ("KET_F_3", "KET_F_7"),
                          ("KET_F_4", "KET_F_8")]

            for old, new in repStrings:
                drugCol = [x.replace(old, new) for x in drugCol]

    for drug_col_i, (drug_col, dataStr) in enumerate(zip(drugCol_Orig, drugCol)):

        densityCol = drug_col.replace(' count', ' density (cells/mm^3)')

        # Pull out the counts and the region columns
        tmpDb = ls_Raw[['Region', drug_col, densityCol]]
        tmpDb = tmpDb.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        drug, sex, dsIdx = dataStr.split('_')
        tmpDb['drug'] = drug
        tmpDb['sex'] = sex
        tmpDb['dataset'] = drug + dsIdx
        tmpDb = tmpDb.rename(columns={drug_col: 'count', densityCol: 'density (cells/mm^3)'})
        tmpDb['total_cells'] = tmpDb[tmpDb['Region'].str.contains('Basic cell groups')]['count'].sum().astype(int)

        if drug_col_i == 0:
            lightSheet_DB = tmpDb
        else:
            lightSheet_DB = pd.concat([lightSheet_DB, tmpDb])

        # remove, prepare data
        background_ROIs = ['background', 'left root', 'right root']
        lightSheet_DB = lightSheet_DB[~lightSheet_DB['Region'].isin(background_ROIs)]

    if switchDict['testSplit']:
        # A list of datasets to artifically shift to a 'new drug' for the sake of testing within drug variability.
        drugDataSet = 'C-SSRI1', 'C-SSRI2', 'DMT1', 'DMT2'
        newDrugName = 'C-SSRI_t', 'C-SSRI_t', 'DMT_t', 'DMT_t'

        # Cycle through each drug in the list above, change it
        for drug_i, drug in enumerate(drugDataSet):
            swapInd = lightSheet_DB['dataset'] == drug
            lightSheet_DB['drug'][swapInd] = newDrugName[drug_i]
    
    if switchDict['debugOutputs']:
        lightSheet_DB.to_csv(dirDict['debugDir'] + f'lightsheet_{dbFileTag}.csv')

    return lightSheet_DB