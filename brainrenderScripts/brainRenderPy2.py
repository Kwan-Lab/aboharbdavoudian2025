# def brainRenderRegionList(regionList, alphaList):
import pickle as pkl
from brainrender import Scene
from vedo import embedWindow, Plotter, show  # <- this will be used to render an embedded scene 
from itkwidgets import view
import numpy as np

f_string_template = "C:\OneDrive\KwanLab\Lightsheet_cFos_Pipeline\\1.scaled_Output\classif\data=count_norm-{drug}-nofeatFilt-nofeatAgg\\featureTrans_PowerTransformer(standardize=False)_RobScal_fSel_{fSelMod}_clf_LogReg(multinom, solver='saga')_CV100\\featureDict.pkl"

# Camera settings for each comparison
cameraSet = []
cameraDMT = {
    'pos': (-11105, -6932, 28053),
    'viewup': (0, -1, 0),
    'clippingRange': (19980, 63713),
    'clippingRange': (30226, 71523),
    'focalPoint': (5091, 3618, -6525),
    'distance': 39614,
}
cameraKet = {
    'pos': (163, -9468, 30539),
    'viewup': (0, -1, 0),
    'clippingRange': (22937, 59660),
    'focalPoint': (5091, 3618, -6525),
    'distance': 39614,
}
cameraSSRI = {
    'pos': (-537, -20702, 24234),
    'viewup': (0, -1, -1),
    'clippingRange': (21960, 60725),
    'focalPoint': (5091, 3618, -6525),
    'distance': 39614,
}

cameraSet.append(cameraDMT)
cameraSet.append(cameraKet)
cameraSet.append(cameraSSRI)

drugComps = ['Trypt', 'KetPsi', 'PsiSSRI']
plotTitles = [
    "Psilocybin vs 5-MeO-DMT",
    "Psilocybin vs Ketamine",
    "Psilocybin vs A-SSRI",
]

# For each drug comparison, both feature selection modalities will be checked for data.
# Each one's list is thresholded at its respective value.
featSelMod = ['SelectKBest(k=30)', 'BorFS']
featSelThres = [95, 50]


filtList = True

for drug, camSet, pltTitle in zip(drugComps, cameraSet, plotTitles):

    # Retrieve both the relevant dictionaries
    allRegion, allCount = [], []
    for fsMod, thres in zip(featSelMod, featSelThres):
        dataDict = dict(drug= drug, fSelMod=fsMod)
        dictPath = f_string_template.format(**dataDict)

        with open(dictPath, 'rb') as f:                 
            [name, featureDict] = pkl.load(f)

        regionList = np.array(list(featureDict.keys()))
        featureCount = np.array(list(featureDict.values()))
        
        if filtList:
            keepInd = featureCount > thres
            regionList = regionList[keepInd]
            featureCount = featureCount[keepInd]
        
        allRegion.append(regionList)
        allCount.append(featureCount)

    # Find elements in both lists
    matching_elements = list(set(allRegion[0]).intersection(allRegion[1]))

    # Extract feature counts for the new list, and turn those to alpha values
    featureIdx = [list(allRegion[1]).index(x) for x in matching_elements]
    featureCount = allCount[1][featureIdx]

    # Convert to variables for plotting
    regionList = matching_elements
    alphaList = np.array(featureCount)/100

    # regionList = ['ACAd', 'ACAv', 'AM', 'AV', 'CA3', 'CL', 'LD']
    # alphaList = [1, 1, 0.4, 0.6, 0.7, 0.6, 0.6]

    embedWindow(None)  # <- this will make your scene popup

    popup_scene = Scene(title=pltTitle)
    alphaList = np.array(alphaList)
    # popup_scene.slice("sagittal")

    for region, alph in zip(regionList, alphaList):
        actorObj = popup_scene.add_brain_region(region, alpha=alph) #, alpha=alph, color='b'
        popup_scene.add_label(actorObj, actorObj.name)

    # Set camera. Adjust the camera as you'd like, hit 'C' and the window terminal is populated with new coordinates
    popup_scene.render(interactive=False, camera=camSet, zoom=1.5)