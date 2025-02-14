#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


n = 8  
bond_length = 1.4  

radius = bond_length / (2 * np.sin(45 / 2))

h_distance = 1.1  

angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

carbon_coords = [(radius * np.cos(theta), radius * np.sin(theta)) for theta in angles]


hydrogen_coords = [(radius * np.cos(theta) - h_distance * np.cos(theta),
                    radius * np.sin(theta) - h_distance * np.sin(theta)) for theta in angles]

print("Carbon Coordinates:")
for i, (x, y) in enumerate(carbon_coords):
    print(f"C{i+1}: ({x:.9f}, {y:.9f})")

print("\nHydrogen Coordinates:")
for i, (x, y) in enumerate(hydrogen_coords):
    print(f"H{i+1}: ({x:.9f}, {y:.9f})")

carbon_x, carbon_y = zip(*carbon_coords)
hydrogen_x, hydrogen_y = zip(*hydrogen_coords)

plt.figure(figsize=(6,6))
plt.scatter(carbon_x, carbon_y, color='black', label="Carbons", zorder=2)
plt.scatter(hydrogen_x, hydrogen_y, color='red', label="Hydrogens", zorder=2)

for i in range(n):
    plt.plot([carbon_x[i], carbon_x[(i+1) % n]], 
             [carbon_y[i], carbon_y[(i+1) % n]], 'black')  # C-C bonds
    plt.plot([carbon_x[i], hydrogen_x[i]], 
             [carbon_y[i], hydrogen_y[i]], 'gray', linestyle="dashed")  #hydrogen


plt.axis("equal")  # Keep proportions correct
plt.show()
