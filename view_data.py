# View the raw AFL_stats data

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

# Get year for data
year = input("Enter year: ")

# Create filename
fname = "AFL_stats/" + year + "_stats.txt"

# Load data
data = pd.read_csv(fname)

# Print stats
print("Dataset shape: " + str(data.shape))
print("First 20 rows:")
print(data.head(20))
print("Summary")
print(data.describe())

