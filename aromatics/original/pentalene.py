#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

n = 5 
bond_length = 1.4 
h_distance = 1.1

radius = bond_length / (2 * np.sin(72 / 2))

angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

carbon_coords = np.array([(radius * np.cos(theta), radius * np.sin(theta)) for theta in angles])

hydrogen_coords = [(radius * np.cos(theta) - h_distance * np.cos(theta),
                    radius * np.sin(theta) - h_distance * np.sin(theta)) for theta in angles[2:]]


C1, C2 = carbon_coords[0], carbon_coords[1]  

midpoint = (C1 + C2) / 2


direction = C2 - C1
direction /= np.linalg.norm(direction)


perp_vector = np.array([-direction[1], direction[0]])  # Rotate 90 degrees


reflected_coords = []
for point in carbon_coords[2:]:
    
    vector_to_point = point - midpoint
   
    distance_along_perp = np.dot(vector_to_point, perp_vector)
    
    reflected_point = point - 2 * distance_along_perp * perp_vector
    reflected_coords.append(reflected_point)

reflected_coords = np.array(reflected_coords)


reflected_hydrogens = []
for point in hydrogen_coords:
    vector_to_point = point - midpoint
    distance_along_perp = np.dot(vector_to_point, perp_vector)
    reflected_h = point - 2 * distance_along_perp * perp_vector
    reflected_hydrogens.append(reflected_h)

reflected_hydrogens = np.array(reflected_hydrogens)


pentalene_coords = np.vstack((carbon_coords, reflected_coords))  
pentalene_hydrogens = np.vstack((hydrogen_coords, reflected_hydrogens))  

print("\nCarbon Coordinates:")
for i, (x, y) in enumerate(pentalene_coords):
    print(f"C{i+1}: ({x:.5f}, {y:.5f})")

print("\nHydrogen Coordinates:")
for i, (x, y) in enumerate(pentalene_hydrogens):
    print(f"H{i+1}: ({x:.5f}, {y:.5f})")


plt.figure(figsize=(6,6))
plt.scatter(*zip(*pentalene_coords), color='black', label="Pentalene Carbons", zorder=2)
plt.scatter(*zip(*pentalene_hydrogens), color='red', label="Hydrogens", zorder=2)

n_total = len(pentalene_coords)
for i in range(n_total):
    plt.plot(*zip(*[pentalene_coords[i], pentalene_coords[(i+1) % n_total]]), 'black')


#Hydrogens

#excluded_indices = {0, 1}  

#for i in range(len(pentalene_hydrogens)):
 #   carbon_index = i+2   # Offset because we skipped the first two carbons
  #  if carbon_index not in excluded_indices:  # Skip unwanted indices
   #     plt.plot(*zip(*[pentalene_coords[carbon_index], pentalene_hydrogens[i]]), 
    #             'gray', linestyle="dashed")

plt.axis("equal")
plt.show()
