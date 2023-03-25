import numpy as np
import random

# Create a sequence of x values
x = np.linspace(20, 80, 100)

# Add some random noise to the y values
y = [xi + random.uniform(-10, 10) for xi in x]

# Write the data to a CSV file
with open('LinearRegression/data.csv', 'w') as f:
    f.write('x,y\n')
    for xi, yi in zip(x, y):
        f.write(f'{xi},{yi}\n')
