# Lightsheet_cFos_Pipeline
A set of scripts used to analyze data from whole brain lightsheet experiments.

## General Organization
- **main.ipynb** is where everything launches from, first few blocks set things up, define paths.
- **functionScripts** directory contains scripts with functions grouped by classify, init, plot, helper.
- **testScripts** directory mostly for things being worked on, isolated from main files.
- generate output folders - add to git ignore to avoid uploading.

## Note on setup
- Requires Python 3.10 or earlier.
- Recommend setting up Conda environment. environment.yml file I use included in base directory. Look here for instructions on using this file to initialize environment - [Link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
- file 'allGeneStructureInfo_allgenes_summary_struct.csv' can not be put on Github because it is to big. Get it from elsewhere and put it into 'Atlas' folder with the same name.

## Current Focus
5/25 - Expanding ML pipelines to accommodate binary classification and looping across models. mostly working in **classifySamplesTest** in testScripts.