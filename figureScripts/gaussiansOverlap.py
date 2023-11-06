import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Gaussian distributions
mean1 = 2
spread1 = .5
color1 = 'green'

mean2 = 5
spread2 = 1
color2 = 'blue'

# Generate data points for the Gaussian distributions
data1 = np.random.normal(mean1, spread1, 1000)
data2 = np.random.normal(mean2, spread2, 1000)

# Create the histogram plots
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, density=True, alpha=0.5, color=color1, label=f'Gaussian 1')
plt.hist(data2, bins=30, density=True, alpha=0.5, color=color2, label=f'Gaussian 2')

# Add vertical dashed red line
plt.axvline(np.max(data1), color='red', linestyle='dashed')

# Remove labels and ticks
plt.xticks([])
plt.yticks([])

plt.xlabel('Feature Importance', fontsize=16)

# Show the plot
plt.show()
