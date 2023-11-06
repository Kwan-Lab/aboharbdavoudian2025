import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random
import os

# Figure dir
figDir = os.path.join(os.getcwd(), 'figures_output')
if not os.path.isdir(figDir):
    os.makedirs(figDir)

# Define colors for the columns
# colors = ['red', 'green', 'blue', 'purple', 'orange', 'gold'] # Full set
colors = ['red', 'green', 'purple', 'orange'] # subset to show feature selection

# Define the number of gradient steps and grid size
num_steps = 6
grid_size = len(colors)  # Modify this list to change the number of columns and colors
color_step = 1.6

shuffleSet = [False, True]
perRow = [True, False]

shuffleSet_name = ['Grid.svg', 'Grid_Shuffle.svg']

for shuffle_switch, fName in zip(shuffleSet, shuffleSet_name):

    # Create a gradient matrix
    gradient_matrix = np.zeros((num_steps, grid_size, 4))
    for i in range(num_steps):
        for j in range(grid_size):
            color_index = min(j, len(colors) - 1)
            gradient_matrix[i, j] = mcolors.to_rgba(colors[color_index], 1.0 - ((i*color_step) + 1) / 10)

    # Shuffle columns if needed
    if shuffle_switch:
        for col_i in np.arange(gradient_matrix.shape[1]):
            random.shuffle(gradient_matrix[:, col_i, 3])

    for row_switch in perRow:
        if row_switch:
            # Create a figure and axis for each row of 'gradient_matrix'
            fig, ax = plt.subplots(num_steps, 1, figsize=(grid_size, num_steps))

            # Plot each row of the gradient matrix as a colored table
            for i in range(num_steps):
                grid_data = gradient_matrix[i,:,:]

                ax[i].imshow(grid_data[None, :,:], aspect='auto')

                # Remove x and y ticks
                ax[i].set_xticks([])
                ax[i].set_yticks([])

                # Draw lines to separate cells
                for j in range(1, grid_size):
                    ax[i].axvline(j - 0.5, color='black', linewidth=1)

            fig.savefig(os.path.join(figDir, f'Rows_{fName}'), format='svg', dpi=1200)

        else:
            # Create the figure and axis
            fig, ax = plt.subplots()

            # Plot the gradient matrix as a colored table
            ax.imshow(gradient_matrix, aspect='auto')

            # Remove x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw lines to separate cells
            for i in range(1, num_steps):
                ax.axhline(i - 0.5, color='black', linewidth=1)
            for j in range(1, grid_size):
                ax.axvline(j - 0.5, color='black', linewidth=1)

            fig.savefig(os.path.join(figDir, fName), format='svg', dpi=1200)