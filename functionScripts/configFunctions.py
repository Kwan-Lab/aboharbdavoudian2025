import numpy as np

def return_heatmapDict():

    heatmapDict = dict()
    heatmapDict['data'] = 'density_norm' #cell_density, count, count_norm, density_norm
    heatmapDict['feature'] = 'abbreviation'
    heatmapDict['blockCount'] = 2
    heatmapDict['logChangeSal'] = False
    heatmapDict['areaBlocks'] = True
    heatmapDict['areaPerBlock'] = 4
    heatmapDict['SortList'] = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']

    return heatmapDict

def setup_figure_settings():
    """
    Set up matplotlib and seaborn figure settings to improve readability and consistency.
    """
    import matplotlib as plt
    import seaborn as sns

    # Set global font size
    plt.rcParams['font.size'] = 4

    # Set seaborn style and remove axis spines
    sns.set_style('ticks')
    sns.despine()

    # Set matplotlib settings
    plt.rcParams.update({
        'font.family': 'Helvetica',
        'svg.fonttype': 'none',
        'savefig.dpi': 300,
        'figure.dpi': 300,
        'xtick.major.pad': 2,
        'ytick.major.pad': 0.5,
        'axes.labelpad': 0,
        'legend.frameon': False,
        'legend.loc': 'upper right',
        'figure.frameon': False,
        'axes.linewidth': 0.75,
        'legend.markerscale': 1,
        'savefig.format': 'svg',
    })

    # Set all the font sizes
    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize'] = plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

    return

def setup_figure_changeFonts(fontSize):
    """
    Set up matplotlib and seaborn figure settings to improve readability and consistency.
    """
    import matplotlib as plt
    import seaborn as sns

    # Set global font size
    plt.rcParams['font.size'] = fontSize

    # Set all the font sizes
    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize'] = plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

    return

def return_classifyDict_default():
    classifyDict = dict()

    # Random state related items for reproducibility
    classifyDict['randSeed'] = 82590    # Used by the CV splitter, per scikit learn 'Best practices'
    classifyDict['randState'] = np.random.RandomState(classifyDict['randSeed'])   # Used for all other random state related items

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

    # Parameters for LO Analyses
    classifyDict['LO_drug'] = ['6-F-DET']

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
        classifyDict['CV_count'] = 100 # Number of folds for cross-validation
    elif classifyDict['CVstrat'] == 'StratKFold':
        classifyDict['CV_count'] = 8 # Number of folds for cross-validation

    # Balance cases by under or over sampling
    classifyDict['balance'] = True

    classifyDict['featurePert'] = 'correlation_dependent' # 'interventional' or 'correlation_dependent'

    classifyDict['crossComp_tagList'] = [f"data={classifyDict['data']}-", f"clf_LogReg(multinom)_CV{classifyDict['CV_count']}"]

    return classifyDict

def return_classifyDict_testing():
    classifyDict = return_classifyDict_default()
    
    # For rapid testing of classifier code
    classifyDict['model_featureSel'] = 'Univar' # 'Univar', 'mutInfo', 'RFE', 'MRMR', 'Fdr', 'Fwe_BH', 'Fwe', 'Boruta', 'None'
    classifyDict['CV_count'] = 10 # Number of folds for cross-validation
    classifyDict['featurePert'] = 'interventional' # 'interventional' or 'correlation_dependent'
    classifyDict['saveLoadswitch'] = True

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