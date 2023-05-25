import numpy as np
import matplotlib.pyplot as plt

# Sample 2D matrix

data = np.array([[4, 6, 8],
                 [2, 7, 3],
                 [5, 2, 9]])

# Define colors for each column
colors = ['red', 'green', 'blue']

# Get the number of rows and columns in the matrix
num_rows, num_cols = data.shape

# Set the width of each bar
bar_width = 0.8 / num_cols

# Set the x coordinates for the bars
x = np.arange(num_rows)

# Plotting the bars
for i in range(num_cols):
    plt.bar(x + (i * bar_width), data[:, i], width=bar_width, color=colors[i])

# Add labels and title
plt.xlabel('Rows')
plt.ylabel('Values')
plt.title('Bar Graph with Distinct Colors')

# Add legend
plt.legend(['Column 1', 'Column 2', 'Column 3'])

# Show the plot
plt.show()
