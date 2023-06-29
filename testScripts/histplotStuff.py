import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
vector1 = np.random.normal(0, 1, 1000)
vector2 = np.random.normal(2, 1, 1000)
vector3 = np.random.normal(-2, 1, 1000)

# Plot the histograms
plt.hist(vector1, bins=30, alpha=0.5, label='Vector 1')
plt.hist(vector2, bins=30, alpha=0.5, label='Vector 2')
plt.hist(vector3, bins=30, alpha=0.5, label='Vector 3')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Multiple Vectors')

# Add a legend
plt.legend()

# Show the plot
plt.show()
