import numpy as np

def return_heatmapDict():

    heatmapDict = dict()
    heatmapDict['data'] = 'cell_density' #cell_density, count, count_norm, density_norm
    heatmapDict['feature'] = 'abbreviation'
    heatmapDict['blockCount'] = 2
    heatmapDict['logChangeSal'] = False
    heatmapDict['areaBlocks'] = True
    heatmapDict['areaPerBlock'] = 4
    heatmapDict['SortList'] = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']

    return heatmapDict

def return_classifyDict_default():
    classifyDict = dict()

    classifyDict['seed'] = 82590
    np.random.seed(seed = classifyDict['seed'])

    classifyDict['featurefilt'] = False # True, False
    classifyDict['filtType'] = 'min' # Min removes the bottom 1%, Max removes the top 99th percentile.

    # Parameters for pivoting the data
    classifyDict['data'] = 'count_norm' #cell_density, count, count_norm, density_norm
    classifyDict['feature'] = 'abbreviation'
    classifyDict['label'] = 'drug' # Defined in hf.create_drugClass_dict()
    # hf.create_drugClass_dict(classifyDict)

    # Parameters for feature scaling and aggregation
    classifyDict['featurefilt'] = False # True, False
    classifyDict['filtType'] = 'min' # Min removes the bottom 1%, Max removes the top 99th percentile.
    classifyDict['featureAgg'] = False
    classifyDict['featureSel_linkage'] = 'average'  # 'average', 'complete', 'single', 'ward' (if euclidean)
    classifyDict['featureSel_distance'] = 'correlation' # 'correlation, 'cosine', 'euclidean'
    classifyDict['cluster_count'] = 100 # Number of clusters to generate. Not used at the moment.
    classifyDict['cluster_thres'] = 0.2 # Anything closer than this is merged into a cluster
    
    # Parameters for Preprocessing and feature selection
    classifyDict['model_featureTransform'] = True # True, False
    classifyDict['model_featureScale'] = True # True, False
    classifyDict['model_featureSel'] = 'Boruta' # 'Univar', 'mutInfo', 'RFE', 'MRMR', 'Fdr', 'Fwe_BH', 'Fwe', 'Boruta', 'None'
    classifyDict['model_featureSel_alpha'] = 0.05 # Used for Fdr, Fwe, and Fwe_BH

    # If Fdr/Fwe/None are not used for feature selection, the number of k feature must be preset
    classifyDict['model_featureSel_mode'] = 'modelPer' # 'gridCV', 'modelPer'
    # classifyDict['model_featureSel_k'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    classifyDict['model_featureSel_k'] = [30]

    # Parameters for classification
    classifyDict['model'] = 'LogRegL2' #'LogRegL2', 'LogRegL1', 'LogRegElastic', 'svm'
    classifyDict['multiclass'] = 'multinomial' # 'ovr', 'multinomial'
    classifyDict['max_iter'] = 100
    classifyDict['CVstrat'] = 'ShuffleSplit' #'StratKFold', 'ShuffleSplit'

    # ParamGrid Features - in instances where gridCV is set to true, these are the parameters that will be tested.
    paramGrid = dict()
    # paramGrid['classif__l1_ratio'] = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]          # used for ElasticNet
    # paramGrid['classif__C'] = [0.001, 0.01, 0.1, 1, 10]                    # used for LogisticRegression
    paramGrid['classif__C'] = [1]                    # used for LogisticRegression
    classifyDict['pGrid'] = paramGrid

    classifyDict['shuffle'] = True
    classifyDict['gridCV'] = False

    classifyDict['saveLoadswitch'] = True

    classifyDict['test_size'] = 1/4
    classifyDict['innerFold'] = 4
    if classifyDict['CVstrat'] == 'ShuffleSplit':
        classifyDict['CV_count'] = 1000 # Number of folds for cross-validation
    elif classifyDict['CVstrat'] == 'StratKFold':
        classifyDict['CV_count'] = 8 # Number of folds for cross-validation
    classifyDict['balance'] = True


    classifyDict['featurePert'] = 'correlation_dependent' # 'interventional' or 'correlation_dependent'

    classifyDict['crossComp_tagList'] = [f"data={classifyDict['data']}-", f"clf_LogReg(multinom)_CV{classifyDict['CV_count']}"]

    return classifyDict

def return_classifyDict_testing():
    # For rapid testing of classifier code - key diff is feature selection via Univar, only 10 CV splits, and 'interventional' style SHAP explanations.
    classifyDict = dict()
    
    classifyDict['seed'] = 82590
    np.random.seed(seed = classifyDict['seed'])

    classifyDict['featurefilt'] = False # True, False
    classifyDict['filtType'] = 'min' # Min removes the bottom 1%, Max removes the top 99th percentile.

    # Parameters for pivoting the data
    classifyDict['data'] = 'count_norm' #cell_density, count, count_norm, density_norm
    classifyDict['feature'] = 'abbreviation'
    classifyDict['label'] = 'drug' # Defined in hf.create_drugClass_dict()
    # check hf.create_drugClass_dict() for options

    # Parameters for feature scaling and aggregation
    classifyDict['featurefilt'] = False # True, False
    classifyDict['filtType'] = 'min' # Min removes the bottom 1%, Max removes the top 99th percentile.
    classifyDict['featureAgg'] = False
    classifyDict['featureSel_linkage'] = 'average'  # 'average', 'complete', 'single', 'ward' (if euclidean)
    classifyDict['featureSel_distance'] = 'correlation' # 'correlation, 'cosine', 'euclidean'
    classifyDict['cluster_count'] = 100 # Number of clusters to generate. Not used at the moment.
    classifyDict['cluster_thres'] = 0.2 # Anything closer than this is merged into a cluster
    
    # Parameters for Preprocessing and feature selection
    classifyDict['model_featureTransform'] = True # True, False
    classifyDict['model_featureScale'] = True # True, False
    classifyDict['model_featureSel'] = 'Univar' # 'Univar', 'mutInfo', 'RFE', 'MRMR', 'Fdr', 'Fwe_BH', 'Fwe', 'Boruta', 'None'
    classifyDict['model_featureSel_alpha'] = 0.05 # Used for Fdr, Fwe, and Fwe_BH

    # If Fdr/Fwe/None are not used for feature selection, the number of k feature must be preset
    classifyDict['model_featureSel_mode'] = 'modelPer' # 'gridCV', 'modelPer'
    # classifyDict['model_featureSel_k'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    classifyDict['model_featureSel_k'] = [30]

    # Parameters for classification
    classifyDict['model'] = 'LogRegL2' #'LogRegL2', 'LogRegL1', 'LogRegElastic', 'svm'
    classifyDict['multiclass'] = 'multinomial' # 'ovr', 'multinomial'
    classifyDict['max_iter'] = 100
    classifyDict['CVstrat'] = 'ShuffleSplit' #'StratKFold', 'ShuffleSplit'

    # ParamGrid Features - in instances where gridCV is set to true, these are the parameters that will be tested.
    paramGrid = dict()
    # paramGrid['classif__l1_ratio'] = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]          # used for ElasticNet
    # paramGrid['classif__C'] = [0.001, 0.01, 0.1, 1, 10]                    # used for LogisticRegression
    paramGrid['classif__C'] = [1]                    # used for LogisticRegression
    classifyDict['pGrid'] = paramGrid

    classifyDict['shuffle'] = True
    classifyDict['gridCV'] = False

    classifyDict['saveLoadswitch'] = True

    classifyDict['test_size'] = 1/4
    classifyDict['innerFold'] = 4
    if classifyDict['CVstrat'] == 'ShuffleSplit':
        classifyDict['CV_count'] = 10 # Number of folds for cross-validation
    elif classifyDict['CVstrat'] == 'StratKFold':
        classifyDict['CV_count'] = 8 # Number of folds for cross-validation
    classifyDict['balance'] = True


    classifyDict['featurePert'] = 'interventional' # 'interventional' or 'correlation_dependent'

    classifyDict['crossComp_tagList'] = [f"data={classifyDict['data']}-", 'PowerTrans_RobScal_fSel_SelectKBest(k=30)_clf_LogReg(multinom)_CV10']

    return classifyDict

def return_plotDict():
    plotDict = dict()

    plotDict['shapForcePlotCount'] = 20
    plotDict['shapSummaryThres'] = 75   # Thres of CV inclusion for a feature to be plotted. Set to None to use shapMaxDisplay instead.
    plotDict['shapMaxDisplay'] = 10     # Number of features to show in Shap Summary. Ignored if shapSummaryThres is not None.

    # Switches which determine what is plotted.
    plotDict['plot_ConfusionMatrix'] = True
    plotDict['plot_PRcurve'] = True
    plotDict['plot_SHAPsummary'] = False
    plotDict['plot_SHAPforce'] = False
    plotDict['featureCorralogram'] = False

    return plotDict