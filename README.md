# Lightsheet_cFos_Pipeline
A set of scripts used to analyze data from whole brain lightsheet experiments.

## General Organization
- **main.ipynb** is where everything launches from. 
- **functionScripts** directory contains scripts with functions grouped by classify, init, plot, helper. Contains **configFunctions** which controls many analysis parameters.
- **figureScripts** directory for scripts used to generate figures for paper. Most now refactored to plot functions, remainder focused on some visualizations, including brainrender.
- **Basline Data** - Data folder with Lightsheet data, Atlas folder with Allen Brain Atlas related files for organization.

## Note on setup
- Requires Python 3.10 or newer.
- Recommend setting up Conda environment. environment.yml file I use included in base directory. Look here for instructions on using this file to initialize environment - [Link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
- file 'allGeneStructureInfo_allgenes_summary_struct.csv' can not be put on Github because it is to big. Get it from elsewhere and put it into 'Atlas' folder with the same name.
