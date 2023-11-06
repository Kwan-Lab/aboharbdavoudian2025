import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random
import os
import pickle as pkl
from os.path import exists, join

# Figure dir
figDir = os.path.join(os.getcwd(), 'figures_output')
if not os.path.isdir(figDir):
    os.makedirs(figDir)


# Identify the paths.
f_string_template = "C:\OneDrive\KwanLab\Lightsheet_cFos_Pipeline\\1.scaled_Output\classif\data=count_norm-{drug}-filtMin\\PowerTrans_RobScal_fSel_BorFS_clf_LogReg(multinom)_CV100\\scoreDict.pkl"

# drugComps = ['PsiDMT', '5HTR', 'PsiDF', 
#              'SSRI', 'PsiMDMA', 'PsiSSRI', 'KetPsi']
# plotTitles = ["Psi vs 5-MeO-DMT", "Psi/5-MeO-DMT vs 6-Fluoro-DET", "Psi vs 6-Fluoro-DET", 
#               "Acute SSRI vs Chronic SSRI", "Psi vs MDMA", "Psi vs Acute SSRI", "Psi vs Ketamine"]


drugComps = ['PsiDMT', '5HTR', 'PsiDF']
plotTitles = ["Psi vs 5-MeO-DMT", "Psi/5-MeO-DMT vs 6-Fluoro-DET", "Psi vs 6-Fluoro-DET"]


meanScores = []

for drugComp in drugComps:
    dataDict = dict(drug= drugComp)
    dictPath = f_string_template.format(**dataDict)

    with open(dictPath, 'rb') as f:                 
        featureDict = pkl.load(f)

    meanScores.append(np.mean(featureDict['scores']))

# Plot the bar plot
plt.figure(figsize=(8, 6))
plt.bar(plotTitles, meanScores, color='royalblue')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 1)

# Annotate each bar with its value
for category, value in zip(drugComps, meanScores):
    plt.text(category, value + 1, str(value))

# Display the plot
plt.tight_layout()
plt.savefig("barplot_1.svg", format='svg', bbox_inches='tight')     
plt.show()