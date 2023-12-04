# Retrieve featureDicts
import glob
import pickle as pkl
import numpy as np
import os
from brainrender import Scene
from vedo import embedWindow, Plotter, show  # <- this will be used to render an embedded scene 
from itkwidgets import view
import pandas as pd
import re
from brainrender.video import VideoMaker
from brainrender.camera import set_camera

# sys.path.append('../functionScripts/')
# import helperFunctions as hf

outMode = 'still' # still, vid

cameraKet = {
    'pos': (163, -9468, 30539),
    'viewup': (0, -1, 0),
    'clippingRange': (22937, 59660),
    'focalPoint': (5091, 3618, -6525),
    'distance': 39614,
}

cameraKet2 = {
    'pos': (4984, -3620, 33782),
    'viewup': (0, -1, 0),
    'clippingRange': (26768, 57715),
    'focalPoint': (5089, 3183, -6581),
    'distance': 40932,
}

cameraKetVid = {
    'pos': (34302, -11155, 20007),
    'viewup': (0, -1, 0),
    'clippingRange': (20102, 68096),
    'focalPoint': (5059, 3895, -6370),
    'distance': 42158,
}

# Search for files starting with 'br_' and ending with '.csv'
file_pattern = os.path.join(os.getcwd(), 'testScripts', 'br_*.csv')
matching_files = glob.glob(file_pattern, recursive=True)

# Print the matching file paths
for csv_path in matching_files:
    # Load the file
    pandadf = pd.read_csv(csv_path)

    # Get the name of the file
    match = re.search(r'br_(.*?)\.csv', csv_path)
    pltTitle = match.group(1)

    # Extract data
    regionList = pandadf['abbreviation']
    regionColor = pandadf['Diff'].apply(lambda x: 'b' if x < 0 else 'r')

    embedWindow(None)  # <- this will make your scene popup

    popup_scene = Scene(title=pltTitle)
    
    # popup_scene.slice("sagittal")

    for region, regCol in zip(regionList, regionColor):
        actorObj = popup_scene.add_brain_region(region, color=regCol) #, alpha=alph, color='b'
        popup_scene.add_label(actorObj, actorObj.name)


    if outMode == 'still':
    # Set camera. Adjust the camera as you'd like, hit 'C' and the window terminal is populated with new coordinates
        popup_scene.render(interactive=False, camera=cameraKet2, zoom=1.5)
        print('done')

    elif outMode == 'vid':
        
        set_camera(popup_scene, cameraKetVid)
        vm = VideoMaker(popup_scene, "./examples", "vid_spin")

        # make a video with the custom make frame function
        # this just rotates the scene
        vm.make_video(azimuth=1.5, duration=15, fps=15)

    print('hold') # Drop a breakpoint here to inspect the scene