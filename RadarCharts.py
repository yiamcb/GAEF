# Define categories and indices
categories = ['AD', 'CN', 'FTD']
indices = [0, 15, -1]

# Set the number of features
num_features = len(X[0, 0, ::22])

# Define angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()

# Close the radar chart
angles += angles[:1]

# Create radar chart for each category
for category, index in zip(categories, indices):
    # Extract feature values
    values = X[index, 0, ::22]

    # Close the radar chart
    values = np.concatenate((values,[values[0]]))

    # Plot radar chart
    plt.polar(angles, values, marker='o', label=category)

# Add legend
plt.legend(loc='upper right')

# Set the title and show the plot
plt.title('Comparison of AD, CN, and FTD')
plt.savefig('radar_plot.pdf')
plt.show()