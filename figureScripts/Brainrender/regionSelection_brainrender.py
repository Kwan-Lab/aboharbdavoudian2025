# Retrieve featureDicts
import glob
import pickle as pkl
import numpy as np
import os, sys, re, glob
from brainrender import Scene
from vedo import embedWindow, Plotter, show  # <- this will be used to render an embedded scene 
from itkwidgets import view
import pandas as pd
from brainrender.video import VideoMaker
from brainrender.camera import set_camera
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Set appropriate directories
helpFxnPath = 'C:\\OneDrive\\KwanLab\\Lightsheet_cFos_Pipeline\\functionScripts\\'      # Folder where helper functions are
file_pattern = os.path.join(os.getcwd(), 'figureScripts//Brainrender//', 'br_*.csv')    # Folder where csv files are, and their format
outputDir = os.path.join(os.getcwd(), 'figureScripts//Brainrender//')                  # Folder where output will be saved

# sys.path.append('../functionScripts/')
sys.path.append(helpFxnPath)
import helperFunctions as hf

# Set parameters
outMode = 'still' # still, vid
labelsFlag = False # Adds Brainrender native labels. Creates seperate svgs for labels if false.

# Set camera parameters
cameraScreenshot = {
    'pos': (4284, -2674, 26722),
    'viewup': (0, -1, 0),
    'clippingRange': (26768, 57715),
    'focalPoint': (3678, 4091, -6418),
    'distance': 33828,
}

cameraVid = {
    'pos': (34302, -11155, 20007),
    'viewup': (0, -1, 0),
    'clippingRange': (20102, 68096),
    'focalPoint': (5059, 3895, -6370),
    'distance': 42158,
}

# Identify files according to prior pattern
matching_files = glob.glob(file_pattern, recursive=True)

colorDict = hf.drug_color_map()

# Print the matching file paths
for csv_path in matching_files:
    # Load the file
    pandadf = pd.read_csv(csv_path)

    # Get the name of the file
    match = re.search(r'br_(.*?)\.csv', csv_path)
    pltTitle = match.group(1)

    # Convert to a list of dicts
    convDict = hf.simplified_name_trans_dict()
    for key, value in convDict.items():
        pltTitle = pltTitle.replace(key, value)

    splitNames = pltTitle.split(' vs ')
    drugColors = [colorDict[drug] for drug in splitNames]
    drugColorDict = dict(zip(splitNames, drugColors))

    # Extract data
    regionList = pandadf['abbreviation']
    # Diff Calculation -> pandadf.iloc[:, 1] - pandadf.iloc[:, 2]
    drug1 = pandadf.iloc[:, 1].name
    drug2 = pandadf.iloc[:, 2].name

    regionColor = pandadf['Diff'].apply(lambda x: drugColors[1] if x < 0 else drugColors[0])

    embedWindow(None)  # <- this will make your scene popup

    # popup_scene = Scene(title=pltTitle)
    popup_scene = Scene()

    if labelsFlag:
        labelTag = '_labels'
    else:
        labelTag = ''

    # popup_scene.slice("sagittal")

    for region, regCol in zip(regionList, regionColor):
        actorObj = popup_scene.add_brain_region(region, color=regCol) #, alpha=alph, color='b'
        if labelsFlag:
            popup_scene.add_label(actorObj, actorObj.name)


    if outMode == 'still':
    # Set camera. Adjust the camera as you'd like, hit 'C' and the window terminal is populated with new coordinates
        popup_scene.render(interactive=False, camera=cameraScreenshot, zoom=1.5)
        print('done')

    elif outMode == 'vid':
        
        set_camera(popup_scene, cameraVid)
        vm = VideoMaker(popup_scene, "./examples", "vid_spin")

        # make a video with the custom make frame function
        # this just rotates the scene
        vm.make_video(azimuth=1.5, duration=15, fps=15)

    popup_scene.screenshot(f'{outputDir}{pltTitle}{labelTag}.png')
    # print('hold') # Drop a breakpoint here to inspect the scene

    # ==================== Printing text to put on the side ====================

    if labelsFlag:
        # Create a figure and axis
        fig, ax = plt.subplots()

        for i, drugText in enumerate(drugColorDict.keys()):
            ax.text(0, 0.3 + (i*0.1), drugText, ha='center', va='center', color=drugColorDict[drugText], fontsize=16, fontweight='bold', linespacing=1.5)

        # Iterate through the lists and plot text with color
        for i, (color, text) in enumerate(zip(regionColor, regionList)):
            ax.text(0, 1.1 - (i*0.05), text, ha='center', va='center', color=color, fontsize=16, fontweight='bold', linespacing=1.5)

        # Remove axes
        ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Save the figure as an SVG file
        plt.savefig(f'{outputDir}/{pltTitle}_Text.svg', format='svg', bbox_inches='tight', pad_inches=0)

        # Display the plot
        print('hold')