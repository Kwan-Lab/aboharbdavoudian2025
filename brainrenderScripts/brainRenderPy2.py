# def brainRenderRegionList(regionList, alphaList):
import pickle as pkl
from brainrender import Scene
from vedo import embedWindow, Plotter, show  # <- this will be used to render an embedded scene 
from itkwidgets import view
import numpy as np

path2Dict = [
# "C:\OneDrive\KwanLab\Lightsheet_cFos_Pipeline\\1.scaled_Output\classif\density-Trypt-nofeatFilt-nofeatAgg\\fSel_BorFS_RobScal_clf_LogReg(multinom, solver='saga')_CV100\\featureDict.pkl", 
# "C:\OneDrive\KwanLab\Lightsheet_cFos_Pipeline\\1.scaled_Output\classif\density-KetPsi-nofeatFilt-nofeatAgg\\fSel_BorFS_RobScal_clf_LogReg(multinom, solver='saga')_CV100\\featureDict.pkl",
"C:\OneDrive\KwanLab\Lightsheet_cFos_Pipeline\\1.scaled_Output\classif\density-Psy_NMDA-nofeatFilt-nofeatAgg\\fSel_BorFS_RobScal_clf_LogReg(multinom, solver='saga')_CV100\\featureDict.pkl"]

for dictPath in path2Dict:

    with open(dictPath, 'rb') as f:                 
        [name, featureDict] = pkl.load(f)

    regionList = np.array(list(featureDict.keys()))
    featureCount = np.array(list(featureDict.values()))
    

    thresh = 50
    filtList = True

    if filtList:
        keepInd = featureCount > thresh
        regionList = regionList[keepInd]
        featureCount = featureCount[keepInd]

    alphaList = np.array(featureCount)/100


    # regionList = ['ACAd', 'ACAv', 'AM', 'AV', 'CA3', 'CL', 'LD']
    # alphaList = [1, 1, 0.4, 0.6, 0.7, 0.6, 0.6]

    embedWindow(None)  # <- this will make your scene popup

    popup_scene = Scene(title=name)
    alphaList = np.array(alphaList)
    # popup_scene.slice("sagittal")

    for region, alph in zip(regionList, alphaList):
        actorObj = popup_scene.add_brain_region(region, alpha=alph) #, alpha=alph, color='b'
        popup_scene.add_label(actorObj, actorObj.name)

    # Set camera
    camera = {
        'pos': (20676, -13581, 36884),
        'viewup': (0, -1, 0),
        'clippingRange': (30226, 71523),
        'pos': (20676, -13581, 36884),
        'clippingRange': (30226, 71523),
    }
    zoom = 1.5

    popup_scene.render(interactive=False, camera=camera, zoom=zoom)
    print('d')